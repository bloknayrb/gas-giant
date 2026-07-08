"""ffmpeg mp4 encode helper (gasgiant.export.video).

build_ffmpeg_cmd is pure and tested without ffmpeg; encode_video_job is driven
against a fake Popen so the poll/yield/cancel behaviour is exercised with no
real subprocess (and no ffmpeg install).
"""

from __future__ import annotations

import pytest

from gasgiant.export import video
from gasgiant.jobs import Progress

# -- build_ffmpeg_cmd (pure) --------------------------------------------------


def test_build_ffmpeg_cmd_arguments(tmp_path):
    pattern = tmp_path / "frames" / "frame_%04d.png"
    out = tmp_path / "out.mp4"
    cmd = video.build_ffmpeg_cmd(pattern, out, fps=24, width=512, height=256)

    assert cmd[0] == "ffmpeg"
    # start at frame 0
    i = cmd.index("-start_number")
    assert cmd[i + 1] == "0"
    # framerate carries the fps
    j = cmd.index("-framerate")
    assert cmd[j + 1] == "24"
    # yuv420p output pixel format
    k = cmd.index("-pix_fmt")
    assert cmd[k + 1] == "yuv420p"
    # even-dimensions guard for yuv420p
    v = cmd.index("-vf")
    assert cmd[v + 1] == "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    # the input pattern and the output path
    m = cmd.index("-i")
    assert cmd[m + 1] == str(pattern)
    assert cmd[-1] == str(out)
    # overwrite without prompting
    assert "-y" in cmd


def test_build_ffmpeg_cmd_fps_varies(tmp_path):
    cmd = video.build_ffmpeg_cmd(tmp_path / "f_%04d.png", tmp_path / "o.mp4", fps=30)
    assert cmd[cmd.index("-framerate") + 1] == "30"


# -- ffmpeg_available ---------------------------------------------------------


def test_ffmpeg_available_true(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert video.ffmpeg_available() is True


def test_ffmpeg_available_false(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    assert video.ffmpeg_available() is False


# -- encode_video_job (fake Popen) --------------------------------------------


class _FakePopen:
    """Stands in for subprocess.Popen: poll() returns None ``pending`` times
    then ``exit_code``; kill()/wait() are recorded."""

    def __init__(self, cmd, *, pending=2, exit_code=0, stdout=None, stderr=None, **kw):
        self.cmd = cmd
        self._left = pending
        self._exit = exit_code
        self.returncode = None
        self.killed = False
        # ffmpeg writes to the caller-provided stderr temp file; keep a handle
        # so a "failure" case can seed a diagnostic tail.
        self._stderr = stderr

    def poll(self):
        if self._left > 0:
            self._left -= 1
            return None
        self.returncode = self._exit
        return self._exit

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


def _install(monkeypatch, factory):
    monkeypatch.setattr(video.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video.subprocess, "Popen", factory)
    monkeypatch.setattr(video.time, "sleep", lambda s: None)  # no real waiting


def test_encode_video_job_polls_and_completes(monkeypatch, tmp_path):
    captured = {}

    def factory(cmd, **kw):
        p = _FakePopen(cmd, pending=3, exit_code=0, **kw)
        captured["proc"] = p
        return p

    _install(monkeypatch, factory)
    out = tmp_path / "seq.mp4"
    progs = list(video.encode_video_job(tmp_path / "frames", out, fps=24, width=8, height=4))

    assert all(isinstance(p, Progress) for p in progs)
    assert progs[-1].message == "video encoded"
    assert any(p.message == "encoding video" for p in progs)  # polled while running
    assert not captured["proc"].killed


def test_encode_video_job_raises_on_nonzero_exit(monkeypatch, tmp_path):
    out = tmp_path / "seq.mp4"
    out.write_bytes(b"partial")  # ffmpeg wrote a partial file before failing

    _install(monkeypatch, lambda cmd, **kw: _FakePopen(cmd, pending=1, exit_code=1, **kw))
    with pytest.raises(RuntimeError, match="ffmpeg exited 1"):
        list(video.encode_video_job(tmp_path / "frames", out, fps=24))
    assert not out.exists()  # partial removed on failure


def test_encode_video_job_missing_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        next(video.encode_video_job(tmp_path / "frames", tmp_path / "o.mp4", fps=24))


def test_encode_video_job_cancel_kills_and_unlinks(monkeypatch, tmp_path):
    captured = {}

    def factory(cmd, **kw):
        p = _FakePopen(cmd, pending=10_000, exit_code=0, **kw)  # never finishes
        captured["proc"] = p
        return p

    _install(monkeypatch, factory)
    out = tmp_path / "seq.mp4"
    out.write_bytes(b"partial")  # in-progress encode target

    job = video.encode_video_job(tmp_path / "frames", out, fps=24)
    assert next(job).message == "encoding video"  # ffmpeg still running
    job.close()  # GeneratorExit -> kill + unlink

    assert captured["proc"].killed
    assert not out.exists()
