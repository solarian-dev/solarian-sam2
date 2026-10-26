"""``track``: the object prompted in each episode, tracked through it with SAM 2.

The offline face of the same :class:`~solarian_sam2.model.Sam2Tracker`
the ``sam2-tracker`` service runs live. Writes ``targets/<camera>.npz`` as
solarian_umi_skill's ``write_targets`` does, which ``load_session_replay_buffer``
trains on once every episode has it:

* ``bbox``: (T, 4) float32 x0, y0, x1, y1 in video pixels, -1 where the object is lost;
* ``mask``: (T, H, W) 0/1 bit-packed along W, with ``mask_width`` = W.

T is the episode's row count. An episode is tracked when it has a prompt, its
own or the default one, so correcting a few episodes is a run with just their
prompts: the others keep the targets they have.
"""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import Future
from itertools import islice
from typing import Callable, Iterable, Iterator, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from solarian.core.recording.archive import targets_bytes, video_member
from solarian.datasets import Cancelled, DatasetOp, Episode, EpisodeOut, OpContext, OpKind, Resources
from solarian.datasets.video import decode
from solarian_sam2.client import Sam2TrackerClient
from solarian.core.fisheye import Box, lit_box
from solarian_sam2.model import CONFIG_IMAGE_SIZE, MODELS, Sam2Tracker, mask_to_bbox

#: The server's bearer token, as Solarian Base passes its endpoints' token; unset for an open server.
TOKEN_ENV = "SOLARIAN_BASE_ENDPOINT_TOKEN"
#: Episodes uploaded ahead of the one tracked, and how many at once: one connection
#: to Modal carries ~0.6 MB/s (8 s an episode), four together ~2.2 MB/s.
UPLOAD_AHEAD = 8
UPLOAD_CONNECTIONS = 4


class Prompt(BaseModel):
    """Where the object is on one frame: a point on it, or a box around it."""

    frame: int = Field(default=0, description="Frame the prompt is on; negative counts from the end.")
    point: tuple[float, float] | None = Field(default=None, description="Pixel x, y on the object.")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="Pixel x0, y0, x1, y1 around the object; wins over point.")

    @model_validator(mode="after")
    def _one_of(self) -> Prompt:
        if self.point is None and self.bbox is None:
            raise ValueError("a prompt needs a point or a bbox")
        return self


class TrackParams(BaseModel):
    camera: str | None = Field(default=None, description="Camera to track on (default: the session's first).")
    prompt: Prompt | None = Field(default=None, description="The prompt of every episode without its own.")
    prompts: dict[int, Prompt] = Field(
        default_factory=dict, description="Per-episode prompts, by episode index; they win over prompt.")
    model: Literal[tuple(MODELS)] = Field(default="tiny", description="SAM 2.1 size.")
    res: int | None = Field(default=CONFIG_IMAGE_SIZE, gt=0, multiple_of=16,
                            description=f"Internal resolution, a multiple of 16; lower is faster (none: "
                                        f"{CONFIG_IMAGE_SIZE}).")
    compile: bool = Field(
        default=False, description="torch.compile the model (CUDA only): faster, but minutes to warm up.")
    server: str | None = Field(
        default=None,
        description="URL of a `solarian run-service sam2-tracker` to track on instead of loading SAM 2 "
                    "here (default $SAM2_TRACKER_URL, which Solarian Base sets to the project's running "
                    "sam2-tracker endpoint); it must run this model and res.")
    mask: bool = Field(default=True, description="Store the mask as well as the box.")
    crop: Literal["auto", "none"] | tuple[int, int, int, int] = Field(
        default="auto",
        description="What of the frame SAM 2 sees: x0, y0, x1, y1 pixels, `auto` for what is lit in the "
                    "session's first episode (a fisheye's circle), or `none`. Targets stay in frame pixels.")


def outline(mask: np.ndarray | None) -> list[list[int]]:
    """The mask's outer and hole rings, each a flat ``[x, y, ...]``, as Solarian Base draws them."""
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, 1.0, True).reshape(-1).tolist() for c in contours if len(c) >= 3]


def prefetched(frames: Iterable[np.ndarray], depth: int = 8) -> Iterator[np.ndarray]:
    """``frames``, decoded on a thread ``depth`` ahead, so decoding overlaps tracking
    (each tracking step waits for the GPU). Closing it stops the thread."""
    out: queue.Queue = queue.Queue(depth)
    stop = threading.Event()
    end = object()

    def fill() -> None:
        try:
            for frame in frames:
                while not stop.is_set():
                    try:
                        out.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        pass
                if stop.is_set():
                    return
            item = end
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer's thread
            item = exc
        while not stop.is_set():
            try:
                out.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    threading.Thread(target=fill, daemon=True, name="decode").start()
    try:
        while (item := out.get()) is not end:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


def iter_track(
    tracker: Sam2Tracker, frames: Iterable[np.ndarray], n: int, prompt: Prompt, crop: Box | None = None,
) -> Iterator[tuple[int, np.ndarray | None]]:
    """``(k, mask)`` for each of the first ``n`` frames, in the order SAM 2 tracks them.

    Forwards from the prompt frame to the end, then re-seeded there and
    backwards to the start: SAM 2 only tracks forwards in time. SAM 2 sees
    only ``crop`` of each frame; the prompt goes in and the masks come out in
    the frame's own pixels, None where the object is lost.
    """
    seed_k = prompt.frame % n if prompt.frame < 0 else min(prompt.frame, n - 1)
    x0, y0, x1, y1 = crop or (0, 0, None, None)
    if prompt.bbox:
        bx0, by0, bx1, by1 = prompt.bbox
        seed = {"bbox": (bx0 - x0, by0 - y0, bx1 - x0, by1 - y0)}
    else:
        px, py = prompt.point
        if crop:
            px, py = min(max(px, x0), x1 - 1), min(max(py, y0), y1 - 1)
        seed = {"points": [(px - x0, py - y0)], "labels": [1]}
    shape = None

    def full(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or crop is None:
            return mask
        out = np.zeros(shape, dtype=bool)
        out[y0:y1, x0:x1] = mask
        return out

    # Frames before the prompt, kept to track backwards (~3.7 MB each, cropped from 1440x1080).
    cache: list[np.ndarray] = []
    seed_frame = None
    for k, frame in enumerate(frames):
        if k >= n:
            break
        shape = frame.shape[:2]
        if crop:
            frame = np.ascontiguousarray(frame[y0:y1, x0:x1])
        if k < seed_k:
            cache.append(frame)
        elif k == seed_k:
            seed_frame = frame
            yield k, full(tracker.seed(frame, **seed))
        else:
            yield k, full(tracker.track(frame))
    if seed_frame is None:
        raise ValueError(f"the video ends before prompt frame {seed_k}")
    if cache:
        tracker.seed(seed_frame, **seed)
        while cache:
            k = len(cache) - 1
            yield k, full(tracker.track(cache.pop()))


def collect(
    results: Iterator[tuple[int, np.ndarray | None]], n: int, size: tuple[int, int], store_mask: bool,
    on_frame: Callable[[int, np.ndarray | None], None] = lambda k, mask: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """``(bbox (n, 4), packed masks or None, frames found)`` from :func:`iter_track`'s results.

    ``size`` is the frame's ``(width, height)``. ``on_frame`` sees each mask
    as it comes; once ``cancelled()`` is true, the next one raises
    :class:`Cancelled` and ``results`` is closed, which stops a remote track.
    """
    width, height = size
    boxes = np.full((n, 4), -1.0, dtype=np.float32)
    masks = np.zeros((n, height, (width + 7) // 8), dtype=np.uint8) if store_mask else None
    found = 0
    try:
        for k, mask in results:
            if cancelled():
                raise Cancelled
            on_frame(k, mask)
            box = None if mask is None else mask_to_bbox(mask)
            if box is None:
                continue
            boxes[k] = box
            if masks is not None:
                masks[k] = np.packbits(mask, axis=-1)
            found += 1
    finally:
        results.close()
    return boxes, masks, found


class Uploader:
    """Uploads episodes' videos to the server before they are tracked.

    Each of ``connections`` threads has its own client: a ``requests.Session``
    is not shared across threads.
    """

    def __init__(self, make_client: Callable[[], Sam2TrackerClient], read: Callable[[int], bytes],
                 connections: int = UPLOAD_CONNECTIONS):
        self._read = read
        self._queue: queue.Queue[tuple[int, Future]] = queue.Queue()
        self._keys: dict[int, Future] = {}
        # Daemons: an op has no teardown, and an upload nobody waits for is not worth a hung exit.
        for _ in range(connections):
            threading.Thread(target=self._work, args=(make_client(),), daemon=True).start()

    def ahead(self, episodes: Iterable[int]) -> None:
        """Queue ``episodes`` not uploaded or queued yet, in order, and drop the
        queued ones not among them: the reviewer has moved on from those."""
        episodes = list(episodes)
        for index, future in list(self._keys.items()):
            if index not in episodes and future.cancel():
                del self._keys[index]
        for index in episodes:
            if index not in self._keys:
                self._keys[index] = future = Future()
                self._queue.put((index, future))

    def key(self, index: int) -> str | None:
        """The uploaded key of ``index``, once its upload in flight ends; None when it is
        not under way, so sending it with the track is sooner, or when it failed."""
        future = self._keys.get(index)
        if future is None or future.cancel():
            self._keys.pop(index, None)
            return None
        try:
            return future.result()
        except Exception:
            return None

    def _work(self, client: Sam2TrackerClient) -> None:
        while True:
            index, future = self._queue.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(client.upload_video(self._read(index)))
            except Exception as exc:
                future.set_exception(exc)


class TrackOp(DatasetOp):
    name = "track"
    title = "Track an object"
    description = """
    Track one object per episode with SAM 2, from a point on it (or a box
    around it) on one frame, forwards to the end and backwards to the start.
    Writes per-frame boxes and masks as training targets. Give one prompt for
    every episode; to correct some, run again with prompts for just those.
    """
    kind = OpKind.MAP
    Params = TrackParams
    resources = Resources(gpu=True, gpu_memory_gb=4)
    produces = ("targets/{camera}.npz",)
    service = "sam2-tracker"
    #: One stateful predictor on one device.
    parallel = False

    def setup(self, params: TrackParams, ctx: OpContext) -> None:
        session = ctx.session
        self.camera = params.camera or session.cameras[0]
        if self.camera not in session.cameras:
            raise ValueError(f"{self.camera!r} is not a camera: {session.cameras}")
        unknown = sorted(set(params.prompts) - set(session.episodes()))
        if unknown:
            raise ValueError(f"prompts for episodes {session.path.name} does not have: {unknown}")
        cam = session.meta["cameras"][self.camera]
        self.width, self.height = int(cam["width"]), int(cam["height"])
        for where, prompt in [("prompt", params.prompt), *((f"prompts[{i}]", p) for i, p in params.prompts.items())]:
            self._check(where, prompt)
        if params.prompt is None and not params.prompts:
            # An online review starts this way: each episode's prompt comes with its redo.
            ctx.log("No prompt yet: episodes are tracked as they get one")
        self.crop = self._crop(params, session)
        ctx.log(f"SAM 2 sees {self.camera} cropped to {self.crop}" if self.crop else f"SAM 2 sees all of {self.camera}")
        self.client = self.tracker = self.uploader = None
        server = params.server or os.environ.get("SAM2_TRACKER_URL")
        if server:
            # Long reads: a compiled server's first episode spends minutes compiling.
            def connect() -> Sam2TrackerClient:
                return Sam2TrackerClient(server, timeout=600, token=os.environ.get(TOKEN_ENV))

            self.client = connect()
            info = self.client.info()
            # The version records model and res as what made its masks.
            served = (info.get("model"), info.get("res") or CONFIG_IMAGE_SIZE)
            if served != (params.model, params.res or CONFIG_IMAGE_SIZE):
                raise ValueError(f"{server} runs SAM 2 {served[0]} at {served[1]}, "
                                 f"not {params.model} at {params.res or CONFIG_IMAGE_SIZE}")
            ctx.log(f"SAM 2 {params.model} at {served[1]} on {server} ({info.get('device')})")
            if info.get("videos"):
                # Reviewing, every episode gets a prompt, in order; a run, those with one.
                tracked = params.prompt is not None or not params.prompts
                self.upload_order = [i for i in session.episodes() if tracked or i in params.prompts]
                self.uploader = Uploader(connect, lambda index: session.read_bytes(video_member(index, self.camera)))
                self.uploader.ahead(self.upload_order[:UPLOAD_AHEAD])
            return
        ctx.log(f"Loading SAM 2 {params.model}")
        self.tracker = Sam2Tracker(params.model, res=params.res, compiled=params.compile)
        uncompiled = " (uncompiled: compile is CUDA-only)" if params.compile and not self.tracker.compiled else ""
        ctx.log(f"SAM 2 {params.model} on {self.tracker.device}{uncompiled}")

    def _crop(self, params: TrackParams, session) -> Box | None:
        if params.crop == "none":
            return None
        if params.crop == "auto":
            first = session.episodes()[0]
            return lit_box(islice(decode(session, video_member(first, self.camera)), 30))
        x0, y0, x1, y1 = params.crop
        if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
            raise ValueError(f"crop {params.crop} is not inside the {self.width}x{self.height} {self.camera} frame")
        return params.crop

    def _check(self, where: str, prompt: Prompt | None) -> None:
        if prompt is None:
            return
        xs, ys = (prompt.bbox[0::2], prompt.bbox[1::2]) if prompt.bbox else ((prompt.point[0],), (prompt.point[1],))
        if not all(0 <= x <= self.width for x in xs) or not all(0 <= y <= self.height for y in ys):
            raise ValueError(f"{where} is outside the {self.width}x{self.height} {self.camera} frame: {prompt}")

    def map(self, episode: Episode, params: TrackParams, ctx: OpContext) -> EpisodeOut | None:
        prompt = params.prompts.get(episode.index, params.prompt)
        if prompt is None:
            return None
        self._check(f"episode {episode.index}'s prompt", prompt)
        n = episode.n_rows
        member = episode.video(self.camera)
        if self.client:
            key = None
            if self.uploader:
                key = self.uploader.key(episode.index)
                self.uploader.ahead([i for i in self.upload_order if i > episode.index][:UPLOAD_AHEAD])
            results = self.client.track_episode(
                episode.session.read_bytes(member), n, prompt.model_dump(), self.crop, key=key)
        else:
            results = iter_track(self.tracker, prefetched(decode(episode.session, member, n)), n, prompt, self.crop)
        boxes, masks, found = collect(
            results, n, (self.width, self.height), params.mask,
            on_frame=lambda k, mask: ctx.preview({"frame": k, "rings": outline(mask)}),
            cancelled=ctx.cancelled,
        )
        if found == 0:
            ctx.log(f"WARNING episode {episode.index}: nothing found at {prompt}; its targets are all -1")
        else:
            ctx.log(f"episode {episode.index}: object on {found}/{n} frames")
        arrays = {"bbox": boxes}
        if masks is not None:
            arrays.update(mask=masks, mask_width=np.array(self.width))
        coverage = {"found": found, "frames": n, "op": self.name, "model": params.model}
        if self.crop:
            coverage["crop"] = list(self.crop)
        if found:
            # A prompt that missed the object segments the table or the whole
            # view: found on every frame, but with a box most of the frame.
            seen = boxes[boxes[:, 0] >= 0]
            area = (seen[:, 2] - seen[:, 0]) * (seen[:, 3] - seen[:, 1]) / (self.width * self.height)
            coverage["box_area"] = round(float(np.median(area)), 4)
        return episode.derive(
            {f"targets/{self.camera}.npz": targets_bytes(**arrays)},
            targets={**episode.meta.get("targets", {}), self.camera: coverage},
        )
