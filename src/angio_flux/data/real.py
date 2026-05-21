"""Real coronary angiography dataset (PNG sequences + DICOM-JSON metadata).

Expected layout:
    <root>/<patient>/<study_dir.DCM>/<study_dir.DCM>.<frame_idx>.png
                                    /<study_dir.DCM>.json

`AngioSequenceDataset` yields:
    {
        "video":   (1, T, H, W) float32 in [0, 1]
        "meta":    dict with frame_time_ms, primary_angle, secondary_angle,
                   pixel_spacing_mm, projection_label (str)
        "path":    absolute path of the study directory
    }
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# DICOM tags we care about (group+element concatenated, hex, uppercase).
TAG_ROWS = "00280010"
TAG_COLS = "00280011"
TAG_NUM_FRAMES = "00280008"
TAG_FRAME_TIME = "00181063"
TAG_PRIMARY_ANGLE = "00181510"
TAG_SECONDARY_ANGLE = "00181511"
TAG_PIXEL_SPACING = "00181164"
TAG_PHOTOMETRIC = "00280004"

_FRAME_RE = re.compile(r"\.(\d+)\.png$")


def _angle_to_projection(primary: float, secondary: float) -> str:
    """Coarse C-arm angle → projection name (RAO/LAO + CRA/CAU)."""
    horiz = "LAO" if primary >= 0 else "RAO"
    vert = "CRA" if secondary >= 0 else "CAU"
    return f"{horiz}{abs(primary):.0f}_{vert}{abs(secondary):.0f}"


def _first(meta: dict, key: str, default=None):
    v = meta.get(key)
    if not v or "Value" not in v or not v["Value"]:
        return default
    return v["Value"][0]


@dataclass
class StudyEntry:
    dir: Path
    json_path: Path
    frame_paths: list[Path]


def discover_studies(root: str | Path) -> list[StudyEntry]:
    """Walk a dataset root and collect (json, frames) per study directory."""
    root = Path(root)
    studies: list[StudyEntry] = []
    for json_path in sorted(root.rglob("*.json")):
        study_dir = json_path.parent
        frames = list(study_dir.glob("*.png"))
        # Sort by numeric frame index in the filename (e.g. ".0.png", ".10.png").
        indexed: list[tuple[int, Path]] = []
        for f in frames:
            m = _FRAME_RE.search(f.name)
            if m:
                indexed.append((int(m.group(1)), f))
        if not indexed:
            continue
        indexed.sort(key=lambda x: x[0])
        studies.append(
            StudyEntry(
                dir=study_dir,
                json_path=json_path,
                frame_paths=[p for _, p in indexed],
            )
        )
    return studies


class AngioSequenceDataset(Dataset):
    """Dataset of cine X-ray angiography sequences (PNG + DICOM-JSON metadata).

    Args:
        root:         dataset root directory
        target_size:  (H, W) resize target (typical: 256 or 384). None → keep native.
        max_frames:   if int, sample / truncate to this many frames (uniform).
        min_frames:   skip studies with fewer than this many frames.
        invert:       if True, invert intensity (MONOCHROME1 → MONOCHROME2 convention).
                      Auto-detected from JSON when False.
        return_uint8: if True, also return raw uint8 tensor (for visualization).
    """

    def __init__(
        self,
        root: str | Path,
        target_size: tuple[int, int] | None = (256, 256),
        max_frames: int | None = None,
        min_frames: int = 6,
        invert: bool = False,
        return_uint8: bool = False,
    ) -> None:
        self.studies = [
            s for s in discover_studies(root) if len(s.frame_paths) >= min_frames
        ]
        self.target_size = target_size
        self.max_frames = max_frames
        self.invert = invert
        self.return_uint8 = return_uint8
        if not self.studies:
            raise RuntimeError(f"no usable studies found under {root}")

    def __len__(self) -> int:
        return len(self.studies)

    def _load_meta(self, study: StudyEntry) -> dict:
        with open(study.json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        primary = float(_first(meta, TAG_PRIMARY_ANGLE, 0.0) or 0.0)
        secondary = float(_first(meta, TAG_SECONDARY_ANGLE, 0.0) or 0.0)
        frame_time = float(_first(meta, TAG_FRAME_TIME, 0.0) or 0.0)
        ps = _first(meta, TAG_PIXEL_SPACING, None)
        if isinstance(ps, list):
            pixel_spacing = float(ps[0])
        else:
            try:
                pixel_spacing = float(ps) if ps is not None else 0.0
            except Exception:
                pixel_spacing = 0.0
        photometric = str(_first(meta, TAG_PHOTOMETRIC, "MONOCHROME2"))
        return {
            "frame_time_ms": frame_time,
            "primary_angle": primary,
            "secondary_angle": secondary,
            "pixel_spacing_mm": pixel_spacing,
            "projection_label": _angle_to_projection(primary, secondary),
            "photometric": photometric,
        }

    def _sample_indices(self, n: int) -> list[int]:
        if self.max_frames is None or n <= self.max_frames:
            return list(range(n))
        idx = np.linspace(0, n - 1, self.max_frames).round().astype(int)
        return idx.tolist()

    def __getitem__(self, i: int) -> dict:
        study = self.studies[i]
        meta = self._load_meta(study)

        indices = self._sample_indices(len(study.frame_paths))
        frames = []
        raw = []
        for k in indices:
            img = Image.open(study.frame_paths[k]).convert("L")
            arr = np.array(img, dtype=np.uint8, copy=True)
            if self.return_uint8:
                raw.append(arr)
            t = torch.from_numpy(arr).float() / 255.0  # (H, W)
            frames.append(t)
        video = torch.stack(frames, dim=0)  # (T, H, W)

        invert = self.invert or (meta["photometric"].upper() == "MONOCHROME1")
        if invert:
            video = 1.0 - video

        if self.target_size is not None:
            h, w = self.target_size
            video = F.interpolate(
                video.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)

        video = video.unsqueeze(0)  # (1, T, H, W)
        out = {"video": video, "meta": meta, "path": str(study.dir)}
        if self.return_uint8:
            out["raw_frames"] = np.stack(raw, axis=0)
        return out


def collate_pad(batch: list[dict]) -> dict:
    """Collate sequences of possibly different T by padding to max T with edge replication."""
    max_t = max(item["video"].shape[1] for item in batch)
    videos = []
    valid = []
    for item in batch:
        v = item["video"]  # (1, T, H, W)
        t = v.shape[1]
        if t < max_t:
            pad = v[:, -1:].expand(-1, max_t - t, -1, -1)
            v = torch.cat([v, pad], dim=1)
        videos.append(v)
        valid.append(t)
    videos = torch.stack(videos, dim=0)  # (B, 1, T, H, W)
    metas = [it["meta"] for it in batch]
    paths = [it["path"] for it in batch]
    return {
        "video": videos,
        "valid_frames": torch.tensor(valid, dtype=torch.long),
        "meta": metas,
        "path": paths,
    }
