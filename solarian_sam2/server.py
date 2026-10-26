"""SAM 2 tracker server: ``solarian run-service sam2-tracker``.

Ported from ``segment-anything-2-real-time/tracker_server.py``. One
:class:`~solarian_sam2.model.Sam2Tracker`, one tracking session: the
predictor is stateful (memory bank, frame index), so every call runs on one
inference thread. Endpoints::

    POST /init  {image_jpeg, points, labels | bbox} -> {bbox, mask_png, inference_ms, session}
    POST /track {image_jpeg, session?}              -> {bbox, mask_png, inference_ms}
    POST /reset {}                                  -> {status}
    POST /track_episode?n=&prompt=&crop= <mp4>      -> {frame, mask_png} per line
    PUT  /videos/<sha256> <mp4>                     -> {key}
    POST /track_episode?n=&prompt=&crop=&video=<sha256>  (no body, an uploaded mp4)

``bbox`` is ``[x0, y0, x1, y1]`` pixels of the frame sent, or ``null`` when
the mask is empty. ``mask_png`` is the 0/1 mask at the size of the frame
sent, as a base64 PNG (a few hundred bytes for a 224 frame), or ``null``
with the box.

Each ``/init`` starts a new ``session``. A ``/track`` that names an older one
is a 409: another client re-seeded the tracker, so the masks would be of its
object.

``/track_episode`` is the ``track`` op's tracking run here on an episode's
video: the server decodes it and streams each frame's mask, in tracking
order, at the video's size, as the op's ``iter_track`` yields it. Until the
stream ends, ``/init`` and ``/track`` are a 409. A failure mid-stream is an
``{"error"}`` line, as the status has gone out with the first.

``/videos`` holds mp4s uploaded ahead of their ``/track_episode``, named by
the sha256 of their bytes. One connection to a hosted server carries ~0.6
MB/s, so an episode's upload took longer than tracking it; the ``track`` op
uploads the next episodes, several at once, while one is tracked. A
``video`` the server does not have (evicted, or a new container) is a 404,
and the client sends the mp4 along instead.
"""

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import cv2
import numpy as np

from solarian.datasets.video import decode_file
from solarian.services import ServiceServer
from solarian.utils.wire import decode_arrays, to_serializable
from solarian_sam2.model import DEFAULT_CHECKPOINT_DIR, DEFAULT_RES, MODELS, Sam2Tracker, mask_to_bbox
from solarian_sam2.op import Prompt, iter_track, prefetched

logger = logging.getLogger(__name__)

VIDEO_KEY = re.compile(r"[0-9a-f]{64}")
#: Bytes of uploaded videos kept, least recently used dropped first; a dataset of 250 episodes is ~1.2 GB.
VIDEO_CACHE_BYTES = 4 << 30


class NoTrackingSession(Exception):
    """``/track`` before ``/init`` or after ``/reset``, or any call while an episode streams."""


def encode_mask_png(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8))
    if not ok:
        raise RuntimeError("cv2.imencode('.png') failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_jpeg(buffer: np.ndarray) -> np.ndarray:
    """The JPEG's pixels as RGB, the order SAM 2 normalises and the ``track`` op decodes."""
    frame = cv2.imdecode(np.asarray(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class VideoCache:
    """mp4s by the sha256 of their bytes, as files under ``root``, at most ``limit`` bytes of them."""

    def __init__(self, root: Path, limit: int = VIDEO_CACHE_BYTES):
        self.root = root
        self.limit = limit
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def put(self, key: str, video: bytes) -> None:
        if hashlib.sha256(video).hexdigest() != key:
            raise ValueError("the video's sha256 is not its key")
        part = self.root / f"{key}.{uuid.uuid4().hex}.part"
        part.write_bytes(video)
        with self._lock:
            part.replace(self.root / key)
            files = sorted((p.stat().st_mtime, p.stat().st_size, p) for p in self.root.iterdir() if VIDEO_KEY.fullmatch(p.name))
            total = sum(size for _, size, _ in files)
            for _, size, path in files:
                if total <= self.limit:
                    break
                path.unlink()
                total -= size

    def get(self, key: str) -> bytes | None:
        with self._lock:
            path = self.root / key
            if not path.exists():
                return None
            os.utime(path)
            return path.read_bytes()


class Sam2TrackerServer(ServiceServer):
    title = "SAM 2 Tracker"
    default_port = 8765

    def __init__(
        self,
        model: str = "small",
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        res: Optional[int] = DEFAULT_RES,
        compiled: bool = False,
    ):
        super().__init__(model=model, checkpoint_dir=checkpoint_dir, res=res, compiled=compiled)
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}; choices: {sorted(MODELS)}")
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.res = res
        self.compiled = compiled
        self.tracker: Sam2Tracker | None = None
        self.tracking_active = False
        self.session = ""
        self.streaming = False
        # Not the working directory: on Modal that is a persistent volume.
        self.videos = VideoCache(Path(tempfile.gettempdir()) / "sam2-tracker-videos")
        # Every tracker call runs on this one thread: it serialises the
        # stateful predictor, and torch.compile's CUDA graphs only replay on
        # the thread that recorded them.
        self._inference_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam2")

    def load(self) -> None:
        self.tracker = self._inference_thread.submit(
            Sam2Tracker, self.model, self.checkpoint_dir, self.res, self.compiled,
        ).result()

    def warm_up(self) -> None:
        self._inference_thread.submit(self.tracker.warm_up).result()

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "res": self.res,
            "compiled": self.compiled,
            "device": None if self.tracker is None else str(self.tracker.device),
            "tracking_active": self.tracking_active,
            # Clients upload ahead only to a server that keeps the videos.
            "videos": True,
        }

    @staticmethod
    def _reply(mask: np.ndarray | None, t0: float) -> Dict[str, Any]:
        bbox = None if mask is None else mask_to_bbox(mask)
        return {
            "bbox": bbox,
            "mask_png": None if bbox is None else encode_mask_png(mask),
            "inference_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def init(self, frame: np.ndarray, points=None, labels=None, bbox=None) -> Dict[str, Any]:
        """Seed on ``points`` (with ``labels``) or ``bbox``, which wins."""
        return self._inference_thread.submit(self._init, frame, points, labels, bbox).result()

    def _init(self, frame: np.ndarray, points, labels, bbox) -> Dict[str, Any]:
        if self.streaming:
            raise NoTrackingSession("The tracker is tracking an episode for another client")
        t0 = time.perf_counter()
        try:
            if bbox is not None:
                mask = self.tracker.seed(frame, bbox=bbox)
            else:
                mask = self.tracker.seed(frame, points=points, labels=labels)
        except Exception:
            self.tracking_active = False
            raise
        self.tracking_active = True
        self.session = uuid.uuid4().hex
        return {**self._reply(mask, t0), "session": self.session}

    def track(self, frame: np.ndarray, session: str | None = None) -> Dict[str, Any]:
        return self._inference_thread.submit(self._track, frame, session).result()

    def _track(self, frame: np.ndarray, session: str | None) -> Dict[str, Any]:
        if self.streaming:
            raise NoTrackingSession("The tracker is tracking an episode for another client")
        if not self.tracking_active:
            raise NoTrackingSession("No active tracking session; call /init first")
        if session is not None and session != self.session:
            raise NoTrackingSession("Another client re-seeded the tracker; call /init again")
        t0 = time.perf_counter()
        return self._reply(self.tracker.track(frame), t0)

    def reset(self) -> None:
        self._inference_thread.submit(setattr, self, "tracking_active", False).result()

    def stream_episode(self, video: bytes, n: int, prompt: Prompt, crop) -> Iterator[str]:
        """``/track_episode``'s lines: each step of the track runs on the inference thread."""
        def run(fn, *args):
            return self._inference_thread.submit(fn, *args).result()

        try:
            run(self._begin_episode)
        except NoTrackingSession as exc:
            yield json.dumps({"error": str(exc)}) + "\n"
            return
        frames = iter_track(self.tracker, prefetched(decode_file(io.BytesIO(video), n)), n, prompt, crop)
        try:
            while (item := run(next, frames, None)) is not None:
                k, mask = item
                yield json.dumps({"frame": k, "mask_png": None if mask is None else encode_mask_png(mask)}) + "\n"
        except Exception as exc:
            logger.exception("track_episode failed")
            yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n"
        finally:
            # Also when the client goes away mid-stream, and the response drops this generator.
            run(frames.close)
            run(setattr, self, "streaming", False)

    def _begin_episode(self) -> None:
        if self.streaming:
            raise NoTrackingSession("The tracker is tracking an episode for another client")
        self.streaming = True
        self.tracking_active = False

    def add_routes(self, app) -> None:
        from fastapi import HTTPException, Request
        from fastapi.responses import StreamingResponse

        server = self

        # Sync handlers: FastAPI runs them on its thread pool, so a frame in
        # flight does not block /health.
        def frame_of(payload: dict) -> np.ndarray:
            try:
                return decode_jpeg(payload["image_jpeg"])
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"bad image_jpeg: {exc}")

        @app.post("/init")
        def init(body: dict):
            payload = decode_arrays(body)
            bbox = payload.get("bbox")
            if bbox is None and len(payload.get("points", ())) == 0:
                raise HTTPException(status_code=400, detail="`points` or `bbox` is required")
            frame = frame_of(payload)
            try:
                result = server.init(frame, payload.get("points"), payload.get("labels", [1]), bbox)
            except NoTrackingSession as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            except Exception as exc:
                logger.exception("init failed")
                raise HTTPException(status_code=500, detail=f"init failed: {exc!r}")
            return to_serializable(result)

        @app.post("/track")
        def track(body: dict):
            payload = decode_arrays(body)
            frame = frame_of(payload)
            try:
                result = server.track(frame, payload.get("session"))
            except NoTrackingSession as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            except Exception as exc:
                logger.exception("track failed")
                raise HTTPException(status_code=500, detail=f"track failed: {exc!r}")
            return to_serializable(result)

        def key_of(key: str) -> str:
            if not VIDEO_KEY.fullmatch(key):
                raise HTTPException(status_code=400, detail="a video is named by the sha256 hex of its bytes")
            return key

        @app.put("/videos/{key}")
        async def put_video(key: str, request: Request):
            try:
                server.videos.put(key_of(key), await request.body())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"key": key}

        @app.post("/track_episode")
        async def track_episode(request: Request, n: int, prompt: str, crop: str = "null", video: str | None = None):
            try:
                parsed = Prompt.model_validate_json(prompt)
                box = json.loads(crop)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"bad prompt or crop: {exc}")
            if video is None:
                mp4 = await request.body()
            else:
                mp4 = server.videos.get(key_of(video))
                if mp4 is None:
                    raise HTTPException(status_code=404, detail=f"no uploaded video {video}")
            return StreamingResponse(
                server.stream_episode(mp4, n, parsed, None if box is None else tuple(box)),
                media_type="application/x-ndjson",
            )

        @app.post("/reset")
        def reset(body: dict):
            server.reset()
            return {"status": "success"}

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", choices=sorted(MODELS), default="small")
        parser.add_argument(
            "--checkpoint-dir",
            default=DEFAULT_CHECKPOINT_DIR,
            help="Directory for the sam2.1_hiera_*.pt files. A missing file is "
            "downloaded here (default $SAM2_CHECKPOINT_DIR, else models/sam2 in the project)",
        )
        parser.add_argument(
            "--res",
            type=int,
            default=DEFAULT_RES,
            help=f"Internal image resolution, a multiple of 16. Lower is faster (default {DEFAULT_RES}).",
        )
        parser.add_argument(
            "--compiled",
            action="store_true",
            help="torch.compile the encoders; the first call is slow.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Sam2TrackerServer":
        if args.res is not None and (args.res <= 0 or args.res % 16):
            raise SystemExit(f"--res must be a positive multiple of 16, got {args.res}")
        return cls(
            model=args.model,
            checkpoint_dir=args.checkpoint_dir,
            res=args.res,
            compiled=args.compiled,
        )


def main(argv: Optional[list] = None) -> None:
    Sam2TrackerServer.main(argv)


if __name__ == "__main__":
    main()
