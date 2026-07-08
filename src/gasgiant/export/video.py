"""Optional ffmpeg mp4 encode of an exported colour-frame sequence.

ffmpeg is discovered at runtime (``shutil.which``) -- it is NOT a hard
dependency; absent it, ``encode_video_job`` raises a clear error and callers
gate the feature off. The encode runs as a POLLING generator: it never calls
``subprocess.run`` (a blocking call would freeze the GUI's per-slice driver
for the whole encode), instead spawning ffmpeg with ``Popen`` and yielding
``Progress`` while polling ``proc.poll()``. Cancellation (generator close /
``GeneratorExit``) kills ffmpeg and removes the partial mp4; a non-zero exit
raises with the captured ffmpeg stderr tail.

``build_ffmpeg_cmd`` is a pure function (no process spawn) so the argument
vector is unit-testable without ffmpeg installed.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from gasgiant.jobs import Progress

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

# ffmpeg's default zero-padded frame index pattern; frame 0 is the first frame
# (mirrors the sequence exporter's frames/frame_0000.png naming).
FRAME_PATTERN = "frame_%04d.png"
_POLL_INTERVAL = 0.05  # seconds between poll()s while ffmpeg runs


def ffmpeg_available() -> bool:
    """True when an ``ffmpeg`` executable is on PATH."""
    return shutil.which("ffmpeg") is not None


def build_ffmpeg_cmd(
    frames_pattern: str | Path,
    out_mp4: str | Path,
    fps: int,
    width: int | None = None,
    height: int | None = None,
) -> list[str]:
    """The ffmpeg argument vector to encode ``frames_pattern`` -> ``out_mp4``.

    Pure (spawns nothing) so it is unit-testable without ffmpeg. ``width`` /
    ``height`` are accepted for callers that know the frame size; the encode
    does not need them because the ``scale=trunc(iw/2)*2:trunc(ih/2)*2`` filter
    forces even dimensions from the actual input size (yuv420p requires even
    width/height, and equirect maps are 2:1 so already even -- but an odd export
    width would otherwise make ffmpeg fail).
    """
    return [
        "ffmpeg",
        "-y",  # overwrite the output without prompting
        "-framerate", str(fps),
        "-start_number", "0",  # frame_0000.png is the first frame
        "-i", str(frames_pattern),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]


def encode_video_job(
    frames_dir: str | Path,
    out_mp4: str | Path,
    fps: int,
    width: int | None = None,
    height: int | None = None,
    *,
    pattern: str = FRAME_PATTERN,
) -> Iterator[Progress]:
    """Encode ``frames_dir/frame_%04d.png`` into ``out_mp4`` via ffmpeg, yielding
    ``Progress`` while polling so the caller's frame loop stays live.

    Raises ``RuntimeError`` if ffmpeg is missing or exits non-zero. On generator
    close (``GeneratorExit``) or any error before completion, ffmpeg is killed
    and the partial mp4 removed -- a finished mp4 (after the final yield) is
    never deleted by a late close.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH; install ffmpeg to export video")
    frames_dir = Path(frames_dir)
    out_mp4 = Path(out_mp4)
    cmd = build_ffmpeg_cmd(frames_dir / pattern, out_mp4, fps, width, height)
    log.info("encoding video via ffmpeg: %s", " ".join(cmd))

    # ffmpeg writes progress to stderr every frame; on a long sequence that can
    # exceed a 64K pipe buffer and deadlock while we poll instead of drain, so
    # capture it to a temp file we read only on failure.
    completed = False
    with tempfile.TemporaryFile() as stderr_buf:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_buf)
        try:
            while proc.poll() is None:
                yield Progress(0, 1, "encoding video")
                time.sleep(_POLL_INTERVAL)
            if proc.returncode != 0:
                stderr_buf.seek(0)
                tail = stderr_buf.read().decode("utf-8", "replace")[-2000:]
                raise RuntimeError(f"ffmpeg exited {proc.returncode}: {tail.strip()}")
            completed = True
            yield Progress(1, 1, "video encoded")
        except BaseException:
            # GeneratorExit (cancel) or any error: stop ffmpeg and drop the
            # partial file. A completed encode (completed=True, e.g. a close
            # right after the final yield) is left intact.
            if proc.poll() is None:
                proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=5)
            if not completed:
                out_mp4.unlink(missing_ok=True)
            raise
