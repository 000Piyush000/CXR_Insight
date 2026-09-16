"""
Patient metadata extraction from chest X-ray filenames / paths.

CheXpert ships images under paths such as:

    CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg

with the actual demographics (age, sex, etc.) living in a separate
``train.csv`` alongside the images, keyed by patient id. Many derivative
/ teaching copies of the dataset instead bake the demographics straight
into the filename, e.g.:

    patient00001_58_Male_White_frontal.jpg
    P00042-F-34-AP-Asian.png

This module supports both: it always extracts what it can from the
filename/path itself (patient id, view/projection), and additionally
looks up age/sex/ethnicity from a CheXpert-style CSV (``data/train.csv``
or ``data/metadata.csv`` by default) when one is present, falling back to
inline filename tokens, and finally to "Unknown" when nothing matches.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

VIEW_PATTERNS = {
    "frontal": re.compile(r"frontal|\bpa\b|\bap\b", re.I),
    "lateral": re.compile(r"lateral|\bll\b|\brl\b", re.I),
}

GENDER_PATTERNS = {
    "Male": re.compile(r"(?:^|[_\-])(male|m)(?:[_\-]|$)", re.I),
    "Female": re.compile(r"(?:^|[_\-])(female|f)(?:[_\-]|$)", re.I),
}

ETHNICITY_TOKENS = [
    "white", "black", "asian", "hispanic", "latino", "native american",
    "pacific islander", "african american", "caucasian", "other",
]

PATIENT_ID_PATTERN = re.compile(r"(patient[\s_\-]?\d+|P\d{3,})", re.I)
AGE_PATTERN = re.compile(r"(?:^|[_\-])(\d{1,3})(?:[_\-]|$)")


@dataclass
class PatientMetadata:
    patient_id: str = "Unknown"
    view: str = "Unknown"
    age: str = "Unknown"
    gender: str = "Unknown"
    ethnicity: str = "Unknown"
    source_filename: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _load_csv_index(csv_path: Path) -> dict:
    """Index a CheXpert-style CSV by patient id for fast lookup."""
    index = {}
    if not csv_path.exists():
        return index
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            path_val = row.get("Path") or row.get("path") or ""
            match = PATIENT_ID_PATTERN.search(path_val)
            pid = match.group(1).lower().replace(" ", "").replace("-", "").replace("_", "") if match else None
            if not pid:
                continue
            index[pid] = {
                "age": row.get("Age") or row.get("age"),
                "gender": row.get("Sex") or row.get("Gender") or row.get("gender"),
                "ethnicity": row.get("Ethnicity") or row.get("Race") or row.get("ethnicity"),
            }
    return index


def _norm_pid(pid: Optional[str]) -> Optional[str]:
    if not pid:
        return None
    return pid.lower().replace(" ", "").replace("-", "").replace("_", "")


def extract_metadata(filename: str, csv_lookup_paths: Optional[list] = None) -> PatientMetadata:
    """
    Extract patient metadata from a chest X-ray filename or full path.

    Args:
        filename: the uploaded filename (or a full CheXpert-style path).
        csv_lookup_paths: optional list of CSV files to consult for
            demographics keyed by patient id (CheXpert train.csv format).

    Returns:
        PatientMetadata with whatever fields could be determined; unknown
        fields are set to "Unknown" rather than raising.
    """
    name = Path(filename).name
    stem = Path(filename).stem
    haystack = str(filename)

    meta = PatientMetadata(source_filename=name)

    pid_match = PATIENT_ID_PATTERN.search(haystack)
    if pid_match:
        meta.patient_id = pid_match.group(1)

    for view, pattern in VIEW_PATTERNS.items():
        if pattern.search(haystack):
            meta.view = view.capitalize()
            break

    for gender, pattern in GENDER_PATTERNS.items():
        if pattern.search(stem):
            meta.gender = gender
            break

    for token in ETHNICITY_TOKENS:
        if re.search(rf"(?:^|[_\-\s]){re.escape(token)}(?:[_\-\s]|$)", stem, re.I):
            meta.ethnicity = token.title()
            break

    for age_match in AGE_PATTERN.finditer(stem):
        candidate = int(age_match.group(1))
        if 0 < candidate <= 120:
            meta.age = str(candidate)
            break

    # Prefer an authoritative CSV lookup (real CheXpert layout) when present.
    csv_lookup_paths = csv_lookup_paths or [
        Path("data/train.csv"),
        Path("data/metadata.csv"),
    ]
    norm_pid = _norm_pid(meta.patient_id) if meta.patient_id != "Unknown" else None
    if norm_pid:
        for csv_path in csv_lookup_paths:
            index = _load_csv_index(Path(csv_path))
            if norm_pid in index:
                row = index[norm_pid]
                if row.get("age"):
                    meta.age = str(row["age"])
                if row.get("gender"):
                    meta.gender = str(row["gender"])
                if row.get("ethnicity"):
                    meta.ethnicity = str(row["ethnicity"])
                break

    return meta
