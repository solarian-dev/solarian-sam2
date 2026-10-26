"""The SAM 2.1 camera predictor, shared by the live server and the ``track`` dataset op.

One core, so the boxes a policy is trained on and the boxes it sees live
come from the same model, prompt semantics and post-processing.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Optional, Sequence

import cv2
import numpy as np

from solarian.utils.models import download_missing, models_dir

logger = logging.getLogger(__name__)

#: Checkpoint file and hydra config per model size, as in ``camera_track.py``.
MODELS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}

#: Official SAM 2.1 weights published with the 092824 release.
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{name}"

#: ``image_size`` of every SAM 2.1 config.
CONFIG_IMAGE_SIZE = 1024

#: Frames :meth:`Sam2Tracker.warm_up` tracks: past the 7 memory frames and 16
#: object pointers SAM 2.1 attends to, after which the shapes stop changing.
WARM_UP_STEPS = 24

#: The ImageNet normalisation SAM 2's predictors apply to RGB in [0, 1].
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

#: The resolution the service runs at unless told otherwise; the op defaults
#: to CONFIG_IMAGE_SIZE. On an RTX 3090 tiny tracks ~90 fps at 512 against ~32 at 1024.
DEFAULT_RES = 512

#: ``SAM2_CHECKPOINT_DIR`` points at an existing checkout's ``checkpoints/``.
DEFAULT_CHECKPOINT_DIR = os.environ.get("SAM2_CHECKPOINT_DIR") or str(models_dir("sam2"))


def ensure_checkpoint(directory: str, filename: str) -> str:
    """Return ``directory/filename``, downloading it when the file is absent."""
    path = os.path.join(os.path.expanduser(directory), filename)
    return str(download_missing(path, CHECKPOINT_URL.format(name=filename), "solarian-sam2"))


def hydra_overrides(res: Optional[int], compiled: bool) -> list[str]:
    overrides: list[str] = []
    if res is not None:
        overrides += [
            f"++model.image_size={res}",
            f"++model.image_encoder.neck.position_encoding.image_size={res}",
        ]
    # The small, base_plus and large configs ship a 32x32 RoPE grid for a 1024
    # input (64x64 features), so memory attention rebuilds ``freqs_cis`` on
    # its first call.
    # Compiled, that rebuild is a CUDA-graph output kept on the module, and
    # the next graph run overwrites it under the tracker.
    feat = (res or CONFIG_IMAGE_SIZE) // 16
    overrides += [
        f"++model.memory_attention.layer.self_attention.feat_sizes=[{feat},{feat}]",
        f"++model.memory_attention.layer.cross_attention.feat_sizes=[{feat},{feat}]",
    ]
    if compiled:
        overrides += [
            "++model.compile_image_encoder=true",
            "++model.compile_memory_encoder=true",
            "++model.compile_memory_attention=true",
        ]
    return overrides


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    """Tightest ``[x0, y0, x1, y1]`` around a boolean mask (x1, y1 exclusive), None if empty."""
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w <= 0 or h <= 0:
        return None
    return [int(x), int(y), int(x + w), int(y + h)]


class Sam2Tracker:
    """One object tracked through a stream of frames, at the resolution they come in.

    Stateful (memory bank, frame index) and not thread-safe: a caller that
    shares one across threads serialises its calls.
    """

    def __init__(
        self,
        model: str = "small",
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        res: Optional[int] = None,
        compiled: bool = False,
    ):
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}; choices: {sorted(MODELS)}")
        if res is not None and (res <= 0 or res % 16):
            raise ValueError(f"res must be a positive multiple of 16, got {res}")
        import torch
        from sam2.build_sam import build_sam2_camera_predictor

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # The compile options are written for CUDA graphs, and compiling is
        # only a speed-up: the masks are the same either way.
        if compiled and self.device.type != "cuda":
            logger.warning("torch.compile is CUDA-only here; running SAM 2 uncompiled on %s", self.device)
            compiled = False
        self.model = model
        self.res = res
        self.compiled = compiled
        checkpoint, config = MODELS[model]
        t0 = time.perf_counter()
        self.predictor = build_sam2_camera_predictor(
            config,
            ensure_checkpoint(checkpoint_dir, checkpoint),
            device=self.device,
            vos_optimized=compiled,
            hydra_overrides_extra=hydra_overrides(res, compiled),
        )
        if self.device.type == "cuda":
            self._mean = torch.tensor(IMG_MEAN, device=self.device)[:, None, None]
            self._std = torch.tensor(IMG_STD, device=self.device)[:, None, None]
            self.predictor.perpare_data = self._prepare_on_device
        logger.info(
            "SAM 2 %s ready on %s in %.1f s%s",
            model, self.device, time.perf_counter() - t0, " (first call autotunes)" if compiled else "",
        )

    def _prepare_on_device(self, img: np.ndarray, image_size: int = CONFIG_IMAGE_SIZE, **_):
        """The predictor's ``perpare_data``, normalised on the GPU: its own does it on
        the CPU in float64, 7-19 ms a frame against a 33 ms budget at 30 fps. The
        resize stays cv2's on uint8, ~1 ms, as torch's bilinear differs enough
        to move the masks."""
        import torch

        height, width = img.shape[:2]
        resized = cv2.resize(img, (image_size, image_size))
        x = torch.from_numpy(resized).to(self.device).permute(2, 0, 1).float()
        return (x / 255.0 - self._mean) / self._std, width, height

    @contextlib.contextmanager
    def _inference(self):
        """Inference mode with the autocast each device is fast at.

        bf16 on CUDA from Ampere on; fp16 before it (a T4), which has no bf16
        tensor cores and tracks several times slower in bf16. None on MPS, where
        fp16 autocast tracks about 5x slower than fp32 (torch 2.8, M-series),
        nor on CPU.
        """
        import torch

        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_properties(self.device).major >= 8 else torch.float16
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                yield
            torch.cuda.synchronize()
        else:
            with torch.inference_mode():
                yield

    @staticmethod
    def _mask(obj_ids, logits) -> np.ndarray | None:
        if obj_ids is None or len(obj_ids) == 0 or logits is None:
            return None
        mask = (logits[0, 0] > 0).detach().cpu().numpy()
        return mask if mask.any() else None

    def seed(
        self,
        frame: np.ndarray,
        points: Sequence[Sequence[float]] | None = None,
        labels: Sequence[int] | None = None,
        bbox: Sequence[float] | None = None,
    ) -> np.ndarray | None:
        """Start tracking the object prompted on ``frame``; its mask, or None.

        Discards the previous object: a new prompt switches target.
        """
        with self._inference():
            # frame_idx 0 + load_first_frame resets the memory bank.
            self.predictor.frame_idx = 0
            self.predictor.load_first_frame(frame)
            _, obj_ids, logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=1,
                points=None if points is None else np.asarray(points, dtype=np.float32),
                labels=None if labels is None else np.asarray(labels, dtype=np.int32),
                bbox=None if bbox is None else np.asarray(bbox, dtype=np.float32),
            )
        return self._mask(obj_ids, logits)

    def warm_up(self, steps: int = WARM_UP_STEPS) -> None:
        """Seed on a synthetic frame and track it ``steps`` frames. Compiled, every
        memory-bank size up to full compiles its own graph: here, not on a user's
        first episode."""
        frame = np.full((1080, 1440, 3), 96, np.uint8)
        frame[440:640, 620:820] = 230
        t0 = time.perf_counter()
        self.seed(frame, bbox=(620, 440, 820, 640))
        for _ in range(steps):
            self.track(frame)
        logger.info("SAM 2 warmed up in %.1f s", time.perf_counter() - t0)

    def track(self, frame: np.ndarray) -> np.ndarray | None:
        """The object's mask on the next frame, or None when it is lost."""
        with self._inference():
            obj_ids, logits = self.predictor.track(frame)
        return self._mask(obj_ids, logits)
