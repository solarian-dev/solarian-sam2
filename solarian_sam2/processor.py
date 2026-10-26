"""The bounding box or mask of a clicked object, as a calculated observation field.

Declared on a camera's ``ModuleObs``::

    bbox = CalculatedField(
        inputs=[MyRobot.wrist_camera.camera.rgb_image],
        processor=Sam2BboxProcessor,
        output=BoundingBoxData(),
        conf=str(service_config_path("bbox_tracker")),
    )

or ``processor=Sam2MaskProcessor, output=MaskData()`` for the mask, and
started with ``controller.start_service("wrist_camera", "bbox",
point=(x, y))``. Until then, and after ``stop_service``, a box is all zeros
and a mask is None. The conf's ``execution_place`` decides whether the relay
or the robot calls the tracker; nothing else changes between the two.

Config::

    execution_place: server     # or robot
    server_url: http://gpu-box:8765
    crop: auto                  # the lit fisheye circle; none, or [x0, y0, x1, y1]
    resize: 224                 # long edge of the crop sent to SAM 2; null sends full size
    jpeg_quality: 90
    timeout: 5.0
    init_timeout: 60.0          # the first call may compile kernels
"""

import logging

import cv2
import numpy as np
import requests

from solarian.services import ServiceProcessor

from solarian_sam2.client import Sam2TrackerClient, Target
from solarian.core.fisheye import Box, lit_box

logger = logging.getLogger(__name__)


def scale_for_long_edge(shape: tuple[int, ...], long_edge: int | None) -> float:
    """Factor that brings the longer side of ``shape`` to ``long_edge``."""
    if long_edge is None:
        return 1.0
    return float(long_edge) / float(max(shape[0], shape[1]))


class Sam2BboxProcessor(ServiceProcessor):
    """Seeds SAM 2 on ``params["point"]`` and follows the object frame to frame.

    A frame where the object is not found keeps the previous box, as the
    UMI eval tracker does: a policy conditioned on the box is better served
    by a stale box than by none.

    SAM 2 sees only ``crop`` of each frame, as the ``track`` op does; the
    click goes in and the target comes out in the frame's own pixels.
    """

    def __init__(self, inputs, output, conf: str | None = None):
        super().__init__(inputs, output, conf)
        cfg = self.cfg or {}
        self.server_url = str(cfg.get("server_url", "http://localhost:8765"))
        crop = cfg.get("crop", "auto")
        # `auto` is found again on each click's frame and kept until the next.
        self.auto_crop = crop == "auto"
        self.crop: Box | None = None if crop in ("auto", "none", None) else tuple(int(v) for v in crop)
        resize = cfg.get("resize", 224)
        self.resize = int(resize) if resize is not None else None
        self.jpeg_quality = int(cfg.get("jpeg_quality", 90))
        self.timeout = float(cfg.get("timeout", 5.0))
        self.init_timeout = float(cfg.get("init_timeout", 60.0))
        # Built on the worker thread: a requests session does not survive
        # the fork that puts a robot-side processor in its module process.
        self._client: Sam2TrackerClient | None = None
        self._clear()

    def _ensure_client(self) -> Sam2TrackerClient:
        if self._client is None:
            self._client = Sam2TrackerClient(
                self.server_url, timeout=self.timeout, jpeg_quality=self.jpeg_quality, token=self.token
            )
        return self._client

    def _view(self, shape: tuple[int, ...]) -> Box:
        """The crop of a frame of ``shape``, or all of it."""
        h, w = shape[:2]
        return self.crop or (0, 0, w, h)

    def _to_send(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """The crop as BGR at ``resize``, and the factor that resize scaled it by."""
        x0, y0, x1, y1 = self._view(frame_rgb.shape)
        view = frame_rgb[y0:y1, x0:x1]
        scale = scale_for_long_edge(view.shape, self.resize)
        bgr = cv2.cvtColor(np.asarray(view, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if scale != 1.0:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(
                bgr,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return bgr, scale

    def _to_sent(self, point, scale: float, shape: tuple[int, ...]) -> tuple[float, float]:
        """A frame-pixel point in the sent image's pixels, held inside the crop."""
        x0, y0, x1, y1 = self._view(shape)
        x = min(max(float(point[0]), x0), x1 - 1)
        y = min(max(float(point[1]), y0), y1 - 1)
        return (x - x0) * scale, (y - y0) * scale

    def _to_source(self, bbox, scale: float, shape: tuple[int, ...]) -> np.ndarray | None:
        if bbox is None:
            return None
        h, w = shape[:2]
        ox, oy = self._view(shape)[:2]
        x0, y0, x1, y1 = (float(v) / scale for v in bbox)
        return np.array(
            [
                int(np.clip(x0 + ox, 0, w)),
                int(np.clip(y0 + oy, 0, h)),
                int(np.clip(x1 + ox, 0, w)),
                int(np.clip(y1 + oy, 0, h)),
            ],
            dtype=np.int32,
        )

    # -- what is kept and published; a subclass that emits more extends these.

    def _clear(self) -> None:
        self._last_bbox = None

    def _keep(self, target: Target, scale: float, shape: tuple[int, ...]) -> None:
        """Keep ``target`` when it was found; a miss leaves the last one."""
        bbox = self._to_source(target.bbox, scale, shape)
        if bbox is not None:
            self._last_bbox = bbox

    def _output(self):
        return self._last_bbox

    def on_start(self, frame, params: dict):
        client = self._ensure_client()
        if self.auto_crop:
            self.crop = lit_box([frame])
            logger.info(
                "%s: SAM 2 sees %s", self.label,
                f"the frame cropped to {self.crop}" if self.crop else "the whole frame",
            )
        bgr, scale = self._to_send(frame)
        point = params["point"]
        target = client.init(bgr, self._to_sent(point, scale, frame.shape), timeout=self.init_timeout)
        self._clear()
        self._keep(target, scale, frame.shape)
        if self._last_bbox is None:
            logger.warning("%s: nothing to track at %s", self.label, tuple(point))
        else:
            logger.info(
                "%s: locked at %s, bbox %s (%.0f ms)",
                self.label,
                tuple(point),
                self._last_bbox.tolist(),
                client.last_inference_ms,
            )
        return self._output()

    def on_frame(self, frame):
        client = self._ensure_client()
        bgr, scale = self._to_send(frame)
        try:
            target = client.track(bgr)
        except requests.HTTPError as exc:
            # 409: the server lost the session (restarted, or another client
            # reset it). Re-seed where the object was last seen.
            if exc.response is None or exc.response.status_code != 409 or self._last_bbox is None:
                raise
            x0, y0, x1, y1 = self._last_bbox.tolist()
            center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            logger.warning("%s: tracker has no session; re-seeding at %s", self.label, center)
            target = client.init(
                bgr, self._to_sent(center, scale, frame.shape), timeout=self.init_timeout
            )
        self._keep(target, scale, frame.shape)
        return self._output()

    def on_stop(self) -> None:
        self._clear()
        self._ensure_client().reset()


class Sam2MaskProcessor(Sam2BboxProcessor):
    """The tracked object's mask at the camera's resolution, for ``MaskData``.

    Same seeding, re-seeding and hold-on-miss as :class:`Sam2BboxProcessor`;
    the mask comes back at the sent crop's size, is scaled up with
    nearest-neighbour so it stays 0/1, and is zero outside the crop.
    """

    def _clear(self) -> None:
        super()._clear()
        self._last_mask = None

    def _keep(self, target: Target, scale: float, shape: tuple[int, ...]) -> None:
        super()._keep(target, scale, shape)
        if target.bbox is not None and target.mask is not None:
            x0, y0, x1, y1 = self._view(shape)
            mask = np.zeros(shape[:2], dtype=np.uint8)
            mask[y0:y1, x0:x1] = cv2.resize(
                target.mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
            )
            self._last_mask = mask

    def _output(self):
        return self._last_mask
