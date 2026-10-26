"""Client for :mod:`solarian_sam2.server`.

Frames travel as JPEG bytes: SAM 2 normalises them anyway, and a
camera-rate stream from a robot over WiFi cannot afford raw pixels.
"""

import base64
import hashlib
import json
from typing import Iterator, NamedTuple, Sequence

import cv2
import numpy as np

import requests

from solarian.services import ServiceClient


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') failed")
    return buf.reshape(-1)


def decode_mask_png(mask_png: str | None) -> np.ndarray | None:
    if mask_png is None:
        return None
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(mask_png), np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError("cv2.imdecode() failed for mask_png")
    return mask


class Target(NamedTuple):
    """The tracked object in the coordinates of the frame sent; both None when lost."""

    #: ``[x0, y0, x1, y1]`` pixels.
    bbox: list[float] | None
    #: ``(H, W)`` uint8 0/1.
    mask: np.ndarray | None


class Sam2TrackerClient(ServiceClient):
    """``init`` / ``track`` / ``reset`` on a BGR frame."""

    def __init__(
        self, server_url: str, timeout: float = 5.0, jpeg_quality: int = 90, token: str | None = None
    ):
        super().__init__(server_url, timeout=timeout, token=token)
        self.jpeg_quality = jpeg_quality
        #: Model time of the last call as the server measured it, in ms.
        self.last_inference_ms = 0.0
        #: What the last ``init`` started; ``track`` names it so a re-seed by another client is a 409.
        self.session: str | None = None

    def _target(self, reply: dict) -> Target:
        self.last_inference_ms = float(reply["inference_ms"])
        # A server older than mask_png sends the box alone.
        return Target(reply["bbox"], decode_mask_png(reply.get("mask_png")))

    def init(
        self,
        frame_bgr: np.ndarray,
        point: tuple[float, float] | None = None,
        timeout: float | None = None,
        bbox: Sequence[float] | None = None,
    ) -> Target:
        """Seed on a point on the object, or a box around it, which wins."""
        prompt = (
            {"bbox": [float(v) for v in bbox]}
            if bbox is not None
            else {"points": [[float(point[0]), float(point[1])]], "labels": [1]}
        )
        reply = self.post(
            "/init",
            {"image_jpeg": encode_jpeg(frame_bgr, self.jpeg_quality), **prompt},
            timeout=timeout,
        )
        # A server older than sessions sends none, and takes any /track.
        self.session = reply.get("session")
        return self._target(reply)

    def track(self, frame_bgr: np.ndarray) -> Target:
        payload = {"image_jpeg": encode_jpeg(frame_bgr, self.jpeg_quality)}
        if self.session is not None:
            payload["session"] = self.session
        return self._target(self.post("/track", payload))

    def reset(self) -> None:
        self.post("/reset", {})

    def upload_video(self, video: bytes) -> str:
        """Keep ``video`` on the server for a later ``track_episode``; its key."""
        key = hashlib.sha256(video).hexdigest()
        response = self._http.put(
            f"{self.server_url}/videos/{key}", data=video, headers={"Content-Type": "video/mp4"}, timeout=self.timeout)
        response.raise_for_status()
        return key

    def track_episode(
        self, video: bytes, n: int, prompt: dict, crop: Sequence[int] | None = None, key: str | None = None,
    ) -> Iterator[tuple[int, np.ndarray | None]]:
        """The ``track`` op's ``iter_track`` run by the server on an episode's mp4.

        Sends the video once (an episode's few MB, against tens of MB of
        frames), or names it by ``key`` when :meth:`upload_video` sent it
        already, and yields ``(k, mask)`` as the server tracks each frame, the
        mask boolean in the video's pixels or None where the object is lost.
        Closing the generator ends the track on the server.
        """
        url = f"{self.server_url}/track_episode"
        params = {"n": n, "prompt": json.dumps(prompt), "crop": json.dumps(None if crop is None else list(crop))}
        response = None
        if key is not None:
            response = self._http.post(url, params={**params, "video": key}, stream=True, timeout=self.timeout)
            if response.status_code == 404:
                response.close()
                response = None
        if response is None:
            response = self._http.post(
                url, params=params, data=video, headers={"Content-Type": "video/mp4"}, stream=True, timeout=self.timeout)
        try:
            if not response.ok:
                raise requests.HTTPError(
                    f"{response.status_code} on /track_episode: {response.text[:300]}", response=response)
            for line in response.iter_lines():
                if not line:
                    continue
                reply = json.loads(line)
                if "error" in reply:
                    raise RuntimeError(f"{self.server_url}: {reply['error']}")
                mask = decode_mask_png(reply["mask_png"])
                yield reply["frame"], None if mask is None else mask.astype(bool)
        finally:
            response.close()
