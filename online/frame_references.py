"""Verify exact decoded source frames independently of the retrieval catalog."""
from __future__ import annotations
import hashlib
import os
import re
from shared.schemas.frame import VerifiedFrameRef
from offline.preprocessing.keyframes import probe_frame_timestamps
from .media import source_video


class SourceFrameVerifier:
    def __init__(self, registry, *, ffprobe: str | None = None):
        self.registry = registry
        self.ffprobe = ffprobe or os.environ.get("AIC_FFPROBE", "ffprobe")
        self._cache = {}

    def clear(self) -> None:
        self._cache.clear()

    def _source(self, video_id: str):
        if re.fullmatch(r"[A-Za-z0-9_-]+", video_id) is None:
            raise ValueError("unsafe video identifier")
        path = source_video(self.registry, video_id)
        if path is None:
            raise ValueError(f"source video unavailable for frame validation: {video_id}")
        stat = path.stat()
        signature = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._cache.get(video_id)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        times = probe_frame_timestamps(path, ffprobe_binary=self.ffprobe)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != signature[1:]:
            raise RuntimeError("source video changed during verification")
        self._cache[video_id] = (signature, digest.hexdigest(), times)
        return digest.hexdigest(), times

    def timestamps(self, video_id: str) -> list[float]:
        return list(self._source(video_id)[1])

    def verify(self, video_id: str, frame_id: int) -> VerifiedFrameRef:
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        digest, times = self._source(video_id)
        if frame_id >= len(times):
            raise ValueError("frame_id is outside the decoded source video")
        return VerifiedFrameRef(video_id=video_id, frame_id=frame_id,
                                pts_time=times[frame_id], source_sha256=digest)

    def validate(self, reference: VerifiedFrameRef) -> float:
        actual = self.verify(reference.video_id, reference.frame_id)
        if actual != reference:
            raise ValueError("verified frame no longer matches source fingerprint or PTS")
        return actual.pts_time
