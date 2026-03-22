from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import SimpleITK as sitk


SUPPORTED_DICOM_SUFFIXES = {".dcm", ".dicom", ""}


def is_dicom_file(path: Path) -> bool:
    """Best-effort check for whether a file is likely a DICOM file."""
    if not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix in {".txt", ".json", ".xml", ".csv", ".jpg", ".jpeg", ".png", ".nii", ".gz", ".zip"}:
        return False

    if suffix in SUPPORTED_DICOM_SUFFIXES:
        return True

    # Some DICOM files have unusual/no suffix. Keep a small content-based fallback.
    try:
        with open(path, "rb") as f:
            header = f.read(132)
        return len(header) >= 132 and header[128:132] == b"DICM"
    except Exception:
        return False


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool = True) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    if overwrite and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def find_candidate_dicom_dirs(root: Path) -> list[Path]:
    """
    Find directories that contain at least one likely DICOM file.
    Return shallowest directories first to reduce duplicates.
    """
    found: list[Path] = []
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        has_dicom = any(is_dicom_file(f) for f in d.iterdir() if f.is_file())
        if has_dicom:
            found.append(d)

    # Keep only shallowest unique branches.
    minimized: list[Path] = []
    for d in sorted(found, key=lambda x: len(x.parts)):
        if not any(parent in d.parents for parent in minimized):
            minimized.append(d)
    return minimized


def read_series_metadata(dicom_dir: Path) -> list[dict[str, Any]]:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        return []

    rows: list[dict[str, Any]] = []
    for idx, series_id in enumerate(series_ids, start=1):
        file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
        if not file_names:
            continue

        # Read first slice metadata only, no pixel load yet.
        first = file_names[0]
        try:
            meta_reader = sitk.ImageFileReader()
            meta_reader.SetFileName(first)
            meta_reader.LoadPrivateTagsOn()
            meta_reader.ReadImageInformation()

            rows.append(
                {
                    "series_index": idx,
                    "series_id": series_id,
                    "source_dir": str(dicom_dir),
                    "num_files": len(file_names),
                    "first_file": first,
                    "modality": meta_reader.GetMetaData("0008|0060") if meta_reader.HasMetaDataKey("0008|0060") else "",
                    "series_description": meta_reader.GetMetaData("0008|103e") if meta_reader.HasMetaDataKey("0008|103e") else "",
                    "study_description": meta_reader.GetMetaData("0008|1030") if meta_reader.HasMetaDataKey("0008|1030") else "",
                    "body_part_examined": meta_reader.GetMetaData("0018|0015") if meta_reader.HasMetaDataKey("0018|0015") else "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "series_index": idx,
                    "series_id": series_id,
                    "source_dir": str(dicom_dir),
                    "num_files": len(file_names),
                    "first_file": first,
                    "modality": "",
                    "series_description": "",
                    "study_description": "",
                    "body_part_examined": "",
                    "warning": str(e),
                }
            )

    return rows


def scan_all_series(extract_dir: Path) -> list[dict[str, Any]]:
    candidate_dirs = find_candidate_dicom_dirs(extract_dir)
    all_series: list[dict[str, Any]] = []
    for dicom_dir in candidate_dirs:
        all_series.extend(read_series_metadata(dicom_dir))
    return all_series


def choose_best_series(series_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not series_list:
        raise RuntimeError("No DICOM series found after extraction.")

    def score(item: dict[str, Any]) -> tuple[int, int, int]:
        modality_ok = 1 if item.get("modality", "").upper() == "CT" else 0
        text = " ".join(
            [
                item.get("series_description", ""),
                item.get("study_description", ""),
                item.get("body_part_examined", ""),
            ]
        ).lower()
        chest_hint = 1 if any(k in text for k in ["chest", "lung", "thorax", "ngực", "phổi"]) else 0
        num_files = int(item.get("num_files", 0))
        return (modality_ok, chest_hint, num_files)

    ranked = sorted(series_list, key=score, reverse=True)
    return ranked[0]


def convert_series_to_nifti(series_info: dict[str, Any], output_path: Path) -> Path:
    dicom_dir = Path(series_info["source_dir"])
    series_id = series_info["series_id"]

    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
    if not file_names:
        raise RuntimeError(f"No file names found for series: {series_id}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))
    return output_path


def build_paths(input_zip: Path, work_dir: Path) -> dict[str, Path]:
    case_name = input_zip.stem.replace(".nii", "")
    case_dir = work_dir / case_name
    return {
        "case_dir": case_dir,
        "extract_dir": case_dir / "extracted",
        "nifti_dir": case_dir / "nifti",
        "report_path": case_dir / "series_report.json",
        "selected_path": case_dir / "selected_series.json",
        "nifti_path": case_dir / "nifti" / "image.nii.gz",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a ZIP of DICOM files, inspect series, and convert the selected series to NIfTI."
    )
    parser.add_argument("--zip", required=True, help="Path to ZIP file containing DICOM folder(s).")
    parser.add_argument(
        "--work-dir",
        default="./work_cases",
        help="Local working directory for extracted files and outputs.",
    )
    parser.add_argument(
        "--series-id",
        default=None,
        help="Optional exact SeriesInstanceUID to convert. If omitted, the script auto-selects the best candidate.",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep extracted folder even if rerun. Default behavior is overwrite.",
    )
    args = parser.parse_args()

    input_zip = Path(args.zip).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    paths = build_paths(input_zip, work_dir)

    print("=" * 70)
    print("1) EXTRACT ZIP")
    print("ZIP:", input_zip)
    print("Extract to:", paths["extract_dir"])
    extract_zip(input_zip, paths["extract_dir"], overwrite=not args.keep_extracted)

    print("=" * 70)
    print("2) SCAN DICOM SERIES")
    series_list = scan_all_series(paths["extract_dir"])
    if not series_list:
        raise RuntimeError("No DICOM series found inside extracted content.")

    paths["report_path"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["report_path"], "w", encoding="utf-8") as f:
        json.dump(series_list, f, indent=2, ensure_ascii=False)

    print(f"Found {len(series_list)} series. Report saved to: {paths['report_path']}")
    for item in series_list:
        print("-" * 70)
        print("Series ID:", item.get("series_id", ""))
        print("Source dir:", item.get("source_dir", ""))
        print("Modality:", item.get("modality", ""))
        print("Body part:", item.get("body_part_examined", ""))
        print("Study description:", item.get("study_description", ""))
        print("Series description:", item.get("series_description", ""))
        print("Num files:", item.get("num_files", 0))

    print("=" * 70)
    print("3) SELECT SERIES")
    if args.series_id:
        matched = [s for s in series_list if s.get("series_id") == args.series_id]
        if not matched:
            raise RuntimeError(f"Series ID not found: {args.series_id}")
        selected = matched[0]
    else:
        selected = choose_best_series(series_list)

    with open(paths["selected_path"], "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print("Selected series:")
    print(json.dumps(selected, indent=2, ensure_ascii=False))

    print("=" * 70)
    print("4) CONVERT DICOM -> NIFTI")
    nifti_path = convert_series_to_nifti(selected, paths["nifti_path"])

    img = sitk.ReadImage(str(nifti_path))
    print("Saved NIfTI:", nifti_path)
    print("Dimension:", img.GetDimension())
    print("Size (x, y, z):", img.GetSize())
    print("Spacing (x, y, z):", img.GetSpacing())
    print("Origin:", img.GetOrigin())
    print("Direction:", img.GetDirection())
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
