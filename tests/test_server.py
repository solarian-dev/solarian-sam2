import hashlib
import json
import os
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from solarian.utils.wire import encode_arrays

from solarian.datasets.video import VideoEncoder

from solarian_sam2.client import Sam2TrackerClient, decode_mask_png
from solarian_sam2.model import hydra_overrides
from solarian_sam2.server import Sam2TrackerServer, VideoCache


class FakeTracker:
    """Records the thread of every call, as CUDA graphs would care about."""

    def __init__(self):
        self.threads = []
        self.track_error: Exception | None = None
        self.mask = np.zeros((8, 8), dtype=bool)
        self.mask[2:4, 1:3] = True

    def seed(self, frame, points=None, labels=None, bbox=None):
        self.threads.append(threading.get_ident())
        return self.mask

    def track(self, frame):
        self.threads.append(threading.get_ident())
        if self.track_error is not None:
            raise self.track_error
        return self.mask


@pytest.fixture
def server():
    s = Sam2TrackerServer(model="tiny")
    s.tracker = FakeTracker()
    s.loaded = True
    return s


def _jpeg():
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return buf.reshape(-1)


def _post(client, path, **payload):
    return client.post(path, json=encode_arrays({"image_jpeg": _jpeg(), **payload}))


@pytest.mark.parametrize("res, feat", [(None, 64), (512, 32)])
def test_rope_grid_matches_the_feature_map(res, feat):
    """A mismatched grid is rebuilt inside the compiled graph and then overwritten."""
    overrides = hydra_overrides(res, compiled=True)
    assert f"++model.memory_attention.layer.self_attention.feat_sizes=[{feat},{feat}]" in overrides
    assert f"++model.memory_attention.layer.cross_attention.feat_sizes=[{feat},{feat}]" in overrides


def test_predictor_calls_share_one_thread(server):
    server.init(np.zeros((8, 8, 3), np.uint8), [[4, 4]], [1])
    caller = threading.Thread(target=server.track, args=(np.zeros((8, 8, 3), np.uint8),))
    caller.start()
    caller.join()
    first, second = server.tracker.threads
    assert first == second != threading.get_ident()


def test_track_without_session_is_409(server):
    with TestClient(server.build_app()) as client:
        reply = _post(client, "/track")
    assert reply.status_code == 409


def test_replies_carry_the_box_and_its_mask(server):
    with TestClient(server.build_app()) as client:
        init = _post(client, "/init", points=[[4, 4]], labels=[1]).json()
        track = _post(client, "/track").json()
    for reply in (init, track):
        assert reply["bbox"] == [1, 2, 3, 4]
        mask = decode_mask_png(reply["mask_png"])
        assert mask.dtype == np.uint8 and mask.shape == (8, 8)
        assert mask.sum() == 4 and mask[2:4, 1:3].all()


def test_a_lost_object_sends_neither_box_nor_mask(server):
    server.tracker.mask = None
    with TestClient(server.build_app()) as client:
        reply = _post(client, "/init", points=[[4, 4]], labels=[1]).json()
    assert reply["bbox"] is None and reply["mask_png"] is None


def test_a_model_error_is_500_with_its_message_not_409(server):
    server.tracker.track_error = RuntimeError("cudagraph replayed on another thread")
    with TestClient(server.build_app()) as client:
        assert _post(client, "/init", points=[[4, 4]], labels=[1]).status_code == 200
        reply = _post(client, "/track")
    assert reply.status_code == 500
    assert "cudagraph replayed on another thread" in reply.json()["detail"]


def test_a_box_seeds_and_a_new_init_turns_away_the_old_session(server):
    with TestClient(server.build_app()) as client:
        first = _post(client, "/init", bbox=[1, 2, 3, 4]).json()["session"]
        assert _post(client, "/track", session=first).status_code == 200
        second = _post(client, "/init", points=[[4, 4]], labels=[1]).json()["session"]
        assert _post(client, "/track", session=first).status_code == 409
        assert _post(client, "/track", session=second).status_code == 200
        assert _post(client, "/track").status_code == 200
        assert _post(client, "/init").status_code == 400


class PromptTracker:
    """A square around the prompt, on frames of any size; keeps what it was given."""

    device = "cuda"

    def __init__(self):
        self.seen = []

    def _square(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=bool)
        x, y = self.at
        mask[int(y) - 2:int(y) + 2, int(x) - 2:int(x) + 2] = True
        return mask

    def seed(self, frame, points=None, labels=None, bbox=None):
        self.at = points[0]
        self.seen.append(frame.shape)
        return self._square(frame)

    def track(self, frame):
        self.seen.append(frame.shape)
        return self._square(frame)


@pytest.fixture
def episode_server(tmp_path):
    s = Sam2TrackerServer(model="small")
    s.tracker = PromptTracker()
    s.videos = VideoCache(tmp_path / "videos")
    s.loaded = True
    return s


@pytest.fixture
def mp4(tmp_path):
    encoder = VideoEncoder(tmp_path / "episode.mp4", 10)
    for k in range(4):
        encoder.write(np.full((32, 48, 3), 40 * k, dtype=np.uint8))
    return encoder.close().read_bytes()


def _stream(client, video, crop=None, **prompt):
    params = {"n": 4, "prompt": json.dumps(prompt), "crop": json.dumps(crop)}
    reply = client.post("/track_episode", params=params, content=video, headers={"Content-Type": "video/mp4"})
    assert reply.status_code == 200
    return [json.loads(line) for line in reply.text.splitlines()]


def test_an_episode_is_tracked_from_its_video_and_streamed_in_tracking_order(episode_server, mp4):
    with TestClient(episode_server.build_app()) as client:
        lines = _stream(client, mp4, crop=[8, 0, 40, 32], frame=2, point=[20, 10])
        assert client.post("/init", json=encode_arrays({"image_jpeg": _jpeg(), "points": [[4, 4]]})).status_code == 200

    assert [line["frame"] for line in lines] == [2, 3, 1, 0]
    assert episode_server.tracker.seen[:5] == [(32, 32, 3)] * 5
    for line in lines:
        mask = decode_mask_png(line["mask_png"])
        assert mask.shape == (32, 48) and mask[8:12, 18:22].all() and mask.sum() == 16


def test_a_second_client_waits_out_the_episode(episode_server, mp4):
    episode_server.streaming = True
    with TestClient(episode_server.build_app()) as client:
        assert _post(client, "/init", points=[[4, 4]], labels=[1]).status_code == 409
        assert _post(client, "/track").status_code == 409
        [line] = _stream(client, mp4, frame=0, point=[4, 4])
    assert "tracking an episode" in line["error"]


def test_an_uploaded_video_is_tracked_by_its_key(episode_server, mp4):
    key = hashlib.sha256(mp4).hexdigest()
    params = {"n": 4, "prompt": json.dumps({"frame": 0, "point": [20, 10]}), "crop": "null"}
    with TestClient(episode_server.build_app()) as client:
        assert client.post("/track_episode", params={**params, "video": key}).status_code == 404
        assert client.put(f"/videos/{key}", content=mp4[:-1]).status_code == 400
        assert client.put("/videos/../../etc", content=mp4).status_code in (400, 404)
        assert client.put(f"/videos/{key}", content=mp4).json() == {"key": key}
        reply = client.post("/track_episode", params={**params, "video": key})
        assert client.get("/info").json()["videos"] is True
    assert [json.loads(line)["frame"] for line in reply.text.splitlines()] == [0, 1, 2, 3]


def test_the_least_recently_used_videos_go_past_the_limit(tmp_path):
    cache = VideoCache(tmp_path, limit=25)
    videos = {name: name.encode() * 10 for name in ("a", "b", "c")}
    keys = {name: hashlib.sha256(video).hexdigest() for name, video in videos.items()}
    cache.put(keys["a"], videos["a"])
    cache.put(keys["b"], videos["b"])
    os.utime(tmp_path / keys["a"], (0, 0))
    os.utime(tmp_path / keys["b"], (1, 1))
    assert cache.get(keys["a"]) == videos["a"], "a get counts as a use"
    cache.put(keys["c"], videos["c"])
    assert [cache.get(keys[name]) is not None for name in "abc"] == [True, False, True]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([keys["a"], keys["c"]])


def test_the_client_names_an_uploaded_video_and_sends_it_when_the_server_lost_it(episode_server, mp4):
    prompt = {"frame": 0, "point": [20, 10]}
    with TestClient(episode_server.build_app()) as http:
        client = Sam2TrackerClient("http://gpu:8765")
        client._http = routed = Routed(http)
        key = client.upload_video(mp4)
        by_key = list(client.track_episode(mp4, 4, prompt, key=key))
        (episode_server.videos.root / key).unlink()
        resent = list(client.track_episode(mp4, 4, prompt, key=key))

    assert routed.bodies == [None, None, mp4]
    assert [k for k, _ in by_key] == [k for k, _ in resent] == [0, 1, 2, 3]


def test_a_bad_episode_ends_its_stream_with_the_error(episode_server, mp4):
    with TestClient(episode_server.build_app()) as client:
        lines = _stream(client, mp4[:100], frame=0, point=[4, 4])
        assert client.post("/track_episode", params={"n": 4, "prompt": "{}"}, content=mp4).status_code == 400
    assert list(lines[-1]) == ["error"]
    assert not episode_server.streaming


class Routed:
    """The client's requests session, answered by a TestClient."""

    def __init__(self, http):
        self.http = http
        self.bodies = []

    def post(self, url, params, stream, timeout, data=None, headers=None):
        self.bodies.append(data)
        reply = self.http.post(url.split(":8765")[1], params=params, content=data, headers=headers)
        reply.ok = reply.is_success
        return reply

    def put(self, url, data, headers, timeout):
        return self.http.put(url.split(":8765")[1], content=data, headers=headers)


def test_the_client_yields_the_episode_s_masks_and_stops_when_closed(episode_server, mp4):
    with TestClient(episode_server.build_app()) as http:
        client = Sam2TrackerClient("http://gpu:8765")
        client._http = Routed(http)
        results = list(client.track_episode(mp4, 4, {"frame": -1, "point": [20, 10]}))
        first = client.track_episode(mp4, 4, {"frame": 0, "point": [20, 10]})
        next(first)
        first.close()

    assert [k for k, _ in results] == [3, 2, 1, 0]
    assert all(mask.dtype == bool and mask.shape == (32, 48) and mask[8:12, 18:22].all() for _, mask in results)
    with pytest.raises(RuntimeError, match="tracking an episode"):
        episode_server.streaming = True
        with TestClient(episode_server.build_app()) as http:
            client._http = Routed(http)
            list(client.track_episode(mp4, 4, {"frame": 0, "point": [1, 1]}))
