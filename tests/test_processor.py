import time

import numpy as np
import pytest
import requests

from solarian.core.data import BoundingBoxData, ImageData, MaskData

from solarian_sam2 import processor as processor_mod
from solarian_sam2.client import Target
from solarian_sam2.processor import Sam2BboxProcessor, Sam2MaskProcessor, scale_for_long_edge
from solarian_sam2.model import mask_to_bbox


class FakeClient:
    """Answers in the coordinates of the frame it was sent, like the server."""

    def __init__(self, *args, **kwargs):
        self.inits = []
        self.tracks = 0
        self.resets = 0
        self.last_inference_ms = 1.0
        self.track_reply = Target(None, None)

    def init(self, frame_bgr, point, timeout=None):
        self.inits.append((frame_bgr.shape, point))
        x, y = point
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        mask[int(y) - 10:int(y) + 10, int(x) - 10:int(x) + 10] = 1
        return Target([x - 10, y - 10, x + 10, y + 10], mask)

    def track(self, frame_bgr):
        self.tracks += 1
        return self.track_reply

    def reset(self):
        self.resets += 1


class Obs:
    """The two things a processor reads off a ModuleObs."""

    @classmethod
    def get_sensor_fields_spec(cls):
        return {}

    @classmethod
    def get_calculated_fields_spec(cls):
        return {}

    def __init__(self, image):
        self.image = image

    def get(self, name, default=None):
        return self.image


@pytest.fixture
def proc(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(processor_mod, "Sam2TrackerClient", lambda *a, **k: fake)
    p = Sam2BboxProcessor(inputs=[ImageData()], output=BoundingBoxData())
    p._input_field_names = ["image"]
    p.fake = fake
    return p


def _frame(h=400, w=800):
    data = ImageData()
    data.value = np.zeros((h, w, 3), dtype=np.uint8)
    data.timestamp = time.time()
    return data


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_scale_for_long_edge():
    assert scale_for_long_edge((400, 800, 3), 224) == pytest.approx(0.28)
    assert scale_for_long_edge((400, 800, 3), None) == 1.0


def test_mask_to_bbox():
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:5, 3:9] = True
    assert mask_to_bbox(mask) == [3, 2, 9, 5]
    assert mask_to_bbox(np.zeros((4, 4), dtype=bool)) is None


def test_idle_field_is_zero_and_never_calls_server(proc):
    out = proc(Obs(_frame()))
    assert out.value.tolist() == [0, 0, 0, 0]
    assert proc.fake.inits == []


def test_click_point_and_box_are_scaled_through_the_resize(proc):
    proc.start({"point": (400, 200)})
    frame = _frame()
    proc(Obs(frame))
    assert _wait(lambda: proc._result is not None)

    (shape, point), = proc.fake.inits
    assert shape[:2] == (112, 224)
    assert point == pytest.approx((112.0, 56.0))
    out = proc(Obs(_frame()))
    assert out.value.dtype == np.int32
    # +/-10 px in the 224-wide frame is +/-35.7 px in the 800-wide one.
    assert out.value.tolist() == [364, 164, 435, 235]
    assert out.timestamp == pytest.approx(frame.timestamp)


def test_lost_object_holds_last_box(proc):
    proc.start({"point": (400, 200)})
    proc(Obs(_frame()))
    assert _wait(lambda: proc._result is not None)
    first = proc._result[0].tolist()
    proc(Obs(_frame()))
    assert _wait(lambda: proc.fake.tracks >= 1)
    time.sleep(0.02)
    assert proc(Obs(_frame())).value.tolist() == first


def test_lost_session_reseeds_at_the_last_box(proc):
    proc.start({"point": (400, 200)})
    proc(Obs(_frame()))
    assert _wait(lambda: proc._result is not None)

    def no_session(frame_bgr):
        response = requests.Response()
        response.status_code = 409
        raise requests.HTTPError("409", response=response)

    proc.fake.track = no_session
    proc(Obs(_frame()))
    assert _wait(lambda: len(proc.fake.inits) == 2)
    # Centre of [364, 164, 435, 235], in the 224-wide frame sent.
    assert proc.fake.inits[1][1] == pytest.approx((399.5 * 0.28, 199.5 * 0.28))


def test_stop_clears_box_and_resets_server(proc):
    proc.start({"point": (400, 200)})
    proc(Obs(_frame()))
    assert _wait(lambda: proc._result is not None)
    proc.stop()
    assert proc(Obs(_frame())).value.tolist() == [0, 0, 0, 0]
    assert _wait(lambda: proc.fake.resets == 1)


@pytest.fixture
def mask_proc(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(processor_mod, "Sam2TrackerClient", lambda *a, **k: fake)
    p = Sam2MaskProcessor(inputs=[ImageData()], output=MaskData())
    p._input_field_names = ["image"]
    p.fake = fake
    return p


def test_idle_mask_is_none(mask_proc):
    assert mask_proc(Obs(_frame())).value is None


def test_mask_is_scaled_to_the_camera_frame(mask_proc):
    mask_proc.start({"point": (400, 200)})
    mask_proc(Obs(_frame()))
    assert _wait(lambda: mask_proc._result is not None)

    out = mask_proc(Obs(_frame())).value
    assert out.dtype == np.uint8 and out.shape == (400, 800)
    assert set(np.unique(out).tolist()) == {0, 1}
    # The 20 px square around (112, 56) in the 224-wide frame, at 1 / 0.28.
    ys, xs = np.nonzero(out)
    assert (xs.min(), xs.max() + 1) == pytest.approx((364, 436), abs=4)
    assert (ys.min(), ys.max() + 1) == pytest.approx((164, 236), abs=4)


def _fisheye(h=400, w=800):
    """Lit from x 200 to 600, black either side, like a circle in a wide frame."""
    data = _frame(h, w)
    data.value[:, 200:600] = 120
    return data


def test_sam2_sees_the_lit_circle_and_targets_come_back_in_frame_pixels(mask_proc):
    mask_proc.resize = None
    mask_proc.start({"point": (400, 200)})
    mask_proc(Obs(_fisheye()))
    assert _wait(lambda: mask_proc._result is not None)

    # lit_box's 8 px margin around x 200..600.
    assert mask_proc.crop == (192, 0, 608, 400)
    (shape, point), = mask_proc.fake.inits
    assert shape[:2] == (400, 416)
    assert point == pytest.approx((208.0, 200.0))
    out = mask_proc(Obs(_fisheye())).value
    assert out.shape == (400, 800)
    ys, xs = np.nonzero(out)
    assert (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) == (390, 410, 190, 210)


def test_lost_object_holds_last_mask_and_stop_clears_it(mask_proc):
    mask_proc.start({"point": (400, 200)})
    mask_proc(Obs(_frame()))
    assert _wait(lambda: mask_proc._result is not None)
    first = mask_proc._result[0].copy()
    mask_proc(Obs(_frame()))
    assert _wait(lambda: mask_proc.fake.tracks >= 1)
    time.sleep(0.02)
    np.testing.assert_array_equal(mask_proc(Obs(_frame())).value, first)

    mask_proc.stop()
    assert mask_proc(Obs(_frame())).value is None
    assert _wait(lambda: mask_proc.fake.resets == 1)


def test_mask_data_round_trips_through_png():
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[5:9, 10:30] = 1
    data = MaskData()
    data.value = mask
    data.compress()
    assert data.compression_codec == "png"
    data.uncompress()
    np.testing.assert_array_equal(data.value, mask)
