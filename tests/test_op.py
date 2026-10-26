import threading
import time

import numpy as np
import pytest
from pydantic import ValidationError

from solarian.datasets import Cancelled
from solarian.core.fisheye import lit_box
from solarian_sam2.op import Prompt, TrackParams, Uploader, collect, iter_track, outline


class FakeTracker:
    """Finds the object at the column a frame's value names; records the order it saw frames."""

    def __init__(self):
        self.calls = []

    def _mask(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=bool)
        x = int(frame[0, 0, 0])
        if x < frame.shape[1]:
            mask[1:3, x:x + 2] = True
            return mask
        return None

    def seed(self, frame, points=None, labels=None, bbox=None):
        self.calls.append(("seed", int(frame[0, 0, 0]), points, bbox))
        return self._mask(frame)

    def track(self, frame):
        self.calls.append(("track", int(frame[0, 0, 0])))
        return self._mask(frame)


def _frames(columns):
    for x in columns:
        frame = np.zeros((4, 16, 3), dtype=np.uint8)
        frame[0, 0, 0] = x
        yield frame


def _track(tracker, frames, n, prompt, store_mask, crop=None, **callbacks):
    return collect(iter_track(tracker, frames, n, prompt, crop), n, (16, 4), store_mask, **callbacks)


def test_tracks_forwards_from_the_prompt_then_backwards_from_it():
    tracker = FakeTracker()
    boxes, masks, found = _track(tracker, _frames([1, 2, 3, 4, 5]), 5, Prompt(frame=2, point=(3, 2)), True)

    assert [c[:2] for c in tracker.calls] == [
        ("seed", 3), ("track", 4), ("track", 5), ("seed", 3), ("track", 2), ("track", 1)]
    assert tracker.calls[0][2] == [(3.0, 2.0)]
    assert found == 5
    assert boxes[0].tolist() == [1, 1, 3, 3] and boxes[4].tolist() == [5, 1, 7, 3]
    assert masks.shape == (5, 4, 2)
    assert np.unpackbits(masks[4], axis=-1, count=16)[1, 5:7].all()


def test_lost_frames_and_rows_past_the_video_stay_minus_one():
    boxes, masks, found = _track(FakeTracker(), _frames([1, 99, 3]), 4, Prompt(bbox=(0, 0, 4, 4)), False)

    assert found == 2 and masks is None
    assert boxes[1].tolist() == [-1] * 4 and boxes[3].tolist() == [-1] * 4


def test_a_negative_frame_counts_from_the_end():
    tracker = FakeTracker()
    _track(tracker, _frames([1, 2, 3]), 3, Prompt(frame=-1, point=(0, 0)), False)
    assert tracker.calls[0][:2] == ("seed", 3)


def test_each_frame_is_reported_as_it_is_tracked():
    seen = []
    _track(FakeTracker(), _frames([1, 2, 3, 4]), 4, Prompt(frame=1, point=(0, 0)), False,
           on_frame=lambda k, mask: seen.append((k, outline(mask))))

    assert [k for k, _ in seen] == [1, 2, 3, 0]
    [ring] = seen[0][1]
    xs, ys = ring[0::2], ring[1::2]
    assert (min(xs), max(xs), min(ys), max(ys)) == (2, 3, 1, 2)


def test_a_cancelled_track_stops_at_the_next_frame():
    seen = []
    with pytest.raises(Cancelled):
        _track(FakeTracker(), _frames([1, 2, 3]), 3, Prompt(point=(0, 0)), False,
               on_frame=lambda k, mask: seen.append(k), cancelled=lambda: bool(seen))
    assert seen == [0]


def test_sam_2_sees_the_crop_and_the_targets_stay_in_frame_pixels():
    tracker = FakeTracker()
    boxes, masks, found = _track(tracker, _frames([1, 2]), 2, Prompt(point=(6, 2)), True, crop=(4, 0, 12, 4))

    assert tracker.calls[0][2] == [(2.0, 2.0)]
    # The cropped frame's corner is the frame's x=4, which _frames leaves 0: the mask is at crop x 0-1.
    assert found == 2 and boxes[0].tolist() == [4, 1, 6, 3]
    assert np.unpackbits(masks[0], axis=-1, count=16)[1:3, 4:6].all()
    off = Prompt(bbox=(0, 0, 8, 4))
    _track(tracker, _frames([1]), 1, off, False, crop=(4, 0, 12, 4))
    assert tracker.calls[-1][3] == (-4, 0, 4, 4)


def test_the_lit_box_is_the_fisheye_circle_unless_it_fills_the_frame():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[0:40, 10:50] = 200
    assert lit_box([frame], margin=2) == (8, 0, 52, 40)
    assert lit_box([np.full((40, 60, 3), 200, dtype=np.uint8)]) is None
    assert lit_box([np.zeros((40, 60, 3), dtype=np.uint8)]) is None


def test_a_prompt_needs_a_point_or_a_box():
    with pytest.raises(ValidationError):
        Prompt(frame=0)
    params = TrackParams.model_validate({"prompts": {"3": {"point": [1, 2]}}})
    assert params.prompts[3].point == (1.0, 2.0)
    assert TrackParams().crop == "auto"
    assert TrackParams.model_validate({"crop": [1, 2, 3, 4]}).crop == (1, 2, 3, 4)


class SlowUploads:
    """``upload_video`` blocks until the test releases that video; records the order they started."""

    def __init__(self):
        self.started = []
        self.release = {}
        self.lock = threading.Lock()

    def upload_video(self, video):
        with self.lock:
            self.started.append(video)
            gate = self.release.setdefault(video, threading.Event())
        assert gate.wait(5)
        return f"key-{video.decode()}"

    def let_go(self, video):
        with self.lock:
            self.release.setdefault(video, threading.Event()).set()


def test_episodes_upload_ahead_a_few_at_once_and_a_track_waits_only_for_one_under_way():
    uploads = SlowUploads()
    uploader = Uploader(lambda: uploads, lambda index: str(index).encode(), connections=2)
    uploader.ahead([0, 1, 2, 3])
    deadline = time.monotonic() + 5
    while len(uploads.started) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    uploader.ahead([1, 2, 3, 4])
    assert sorted(uploads.started) == [b"0", b"1"], "two connections, in order, each episode once"
    uploader.ahead([0, 1, 3, 4])
    assert 2 not in uploader._keys, "queued, then out of the window: dropped"

    assert uploader.key(4) is None, "not started: sent with its track instead"
    assert uploader.key(7) is None, "never queued"
    threading.Timer(0.05, uploads.let_go, [b"0"]).start()
    assert uploader.key(0) == "key-0", "under way: waited for"
    uploads.let_go(b"1")
    assert uploader.key(1) == "key-1"
    uploads.let_go(b"3")
    assert uploader.key(3) == "key-3"
    assert b"2" not in uploads.started and b"4" not in uploads.started
