import urllib.request

import pytest

from solarian_sam2.model import ensure_checkpoint

FILENAME = "sam2.1_hiera_small.pt"


def test_existing_checkpoint_is_not_downloaded(tmp_path, monkeypatch):
    path = tmp_path / FILENAME
    path.write_bytes(b"already")

    def urlopen(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert ensure_checkpoint(str(tmp_path), FILENAME) == str(path)


def test_missing_checkpoint_is_downloaded(tmp_path, monkeypatch):
    body = b"weights"

    class Response:
        def read(self, n=-1):
            nonlocal body
            chunk, body = body, b""
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    opened = {}

    def urlopen(request, timeout=None):
        opened["url"] = request.full_url
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    path = ensure_checkpoint(str(tmp_path), FILENAME)
    assert path == str(tmp_path / FILENAME)
    assert open(path, "rb").read() == b"weights"
    assert opened["url"].endswith("/" + FILENAME)
    assert not (tmp_path / (FILENAME + ".partial")).exists()


def test_failed_download_leaves_no_partial(tmp_path, monkeypatch):
    class Response:
        def read(self, n=-1):
            raise OSError("dropped")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(OSError, match="dropped"):
        ensure_checkpoint(str(tmp_path), FILENAME)
    assert list(tmp_path.iterdir()) == []
