"""
Pipeline: DICOM ZIP → Detect Nodules → Predict Malignancy → JSON Output

Usage:
    python pipeline.py --zip path/to/dicom.zip
    python pipeline.py --zip path/to/dicom.zip --output results.json --work-dir ./work_cases
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pydicom

from concat_3d import (
    extract_zip,
    scan_all_series,
    choose_best_series,
    convert_series_to_nifti,
    build_paths,
)
from extract_nodule import extract_nodule_with_metadata
from use_monai_detect import run_detection
from predictor import predictor_main


# ──────────────────────────────────────────────
# DICOM metadata helpers
# ──────────────────────────────────────────────

def _extract_patient_info(dicom_file: str) -> dict:
    """
    Đọc tuổi (Age) và giới tính (Gender) từ DICOM file bằng pydicom.
    """
    ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)

    # --- Age ---
    raw_age = ds.get("PatientAge", "N/A")
    age = None
    if raw_age and raw_age != "N/A":
        raw_age = str(raw_age).strip()
        match = re.match(r"(\d+)", raw_age)
        if match:
            age = int(match.group(1))

    # --- Gender ---
    raw_sex = ds.get("PatientSex", "N/A")
    gender = None
    if raw_sex and raw_sex != "N/A":
        raw_sex = str(raw_sex).strip().upper()
        if raw_sex.startswith("M"):
            gender = "Male"
        elif raw_sex.startswith("F"):
            gender = "Female"

    return {"age": age, "gender": gender}


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run_pipeline(zip_path: str, work_dir: str = "./work_cases",
                 output_path: str = "results.json",
                 series_id: str | None = None) -> list[dict]:
    """
    1. Extract ZIP → convert DICOM to NIfTI
    2. Extract patient age & gender from DICOM header
    3. Run MONAI detection on NIfTI
    4. For each detected nodule, extract 3D block → predict malignancy
    5. Write JSON results
    """
    input_zip = Path(zip_path).expanduser().resolve()
    work = Path(work_dir).expanduser().resolve()
    paths = build_paths(input_zip, work)

    # ── Step 1: DICOM → NIfTI ──────────────────────────────
    print("=" * 70)
    print("STEP 1: EXTRACT ZIP & CONVERT TO NIFTI")
    extract_zip(input_zip, paths["extract_dir"], overwrite=True)

    series_list = scan_all_series(paths["extract_dir"])
    if not series_list:
        raise RuntimeError("No DICOM series found inside the ZIP.")

    if series_id:
        matched = [s for s in series_list if s.get("series_id") == series_id]
        if not matched:
            raise RuntimeError(f"Series ID not found: {series_id}")
        selected = matched[0]
    else:
        selected = choose_best_series(series_list)

    series_uid = selected.get("series_id", "unknown")
    print(f"Selected series: {series_uid}")

    nifti_path = convert_series_to_nifti(selected, paths["nifti_path"])
    print(f"NIfTI saved to: {nifti_path}")

    # ── Step 2: Extract patient info from DICOM ────────────
    print("=" * 70)
    print("STEP 2: EXTRACT PATIENT INFO (AGE, GENDER)")
    first_dcm = selected.get("first_file", "")
    patient_info = _extract_patient_info(first_dcm)
    age = patient_info["age"]
    gender = patient_info["gender"]
    print(f"Patient Age : {age}")
    print(f"Patient Gender: {gender}")

    if age is None:
        print("WARNING: Không tìm thấy tuổi trong DICOM, mặc định = 60")
        age = 60
    if gender is None:
        print("WARNING: Không tìm thấy giới tính trong DICOM, mặc định = Male")
        gender = "Male"

    # ── Step 3: Run MONAI detection ────────────────────────
    print("=" * 70)
    print("STEP 3: DETECT NODULES (MONAI RetinaNet)")
    detect_json = str(paths["case_dir"] / "detect_result.json")
    detections = run_detection(
        nifti_path_arg=str(nifti_path),
        output_json_arg=detect_json,
    )
    print(f"Detected {len(detections)} nodule(s)")

    if not detections:
        print("Không phát hiện nodule nào. Pipeline kết thúc.")
        empty_result: list[dict] = []
        _save_json(empty_result, output_path)
        return empty_result

    # ── Step 4: Extract & Predict each nodule ──────────────
    print("=" * 70)
    print("STEP 4: EXTRACT NODULE BLOCKS & PREDICT MALIGNANCY")

    # Đọc NIfTI gốc (non-resampled) bằng mha_path cho extract_nodule
    # extract_nodule_with_metadata cần file gốc với spacing gốc
    mha_path = str(nifti_path)

    results = []

    for det in detections:
        cx, cy, cz = det["center_world"]
        nodule_id = det["nodule_id"]
        print(f"\n--- Nodule #{nodule_id} ---")
        print(f"Center world: ({cx:.2f}, {cy:.2f}, {cz:.2f})")

        # 4a. Extract 3D block
        center_coord = (cx, cy, cz)
        try:
            image_block = extract_nodule_with_metadata(
                mha_path, center_coord, target_shape=(64, 128, 128)
            )
        except Exception as e:
            print(f"[ERROR] Cannot extract nodule #{nodule_id}: {e}")
            continue

        # 4b. Predict malignancy
        t_start = time.time()
        pred_result = predictor_main(image_block, age, gender)
        t_elapsed_ms = int((time.time() - t_start) * 1000)

        results.append({
            "seriesInstanceUID": series_uid,
            "probability": round(pred_result["probability"], 10),
            "predictionLabel": pred_result["predictionLabel"],
            "processingTimeMs": t_elapsed_ms,
            "CoordX": round(cx, 2),
            "CoordY": round(cy, 2),
            "CoordZ": round(cz, 2),
        })

    # ── Step 5: Save results ───────────────────────────────
    print("=" * 70)
    print("STEP 5: SAVE RESULTS")
    _save_json(results, output_path)
    print(f"Results saved to: {output_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    return results


def _save_json(data: list[dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect lung nodules and predict malignancy from a DICOM ZIP."
    )
    parser.add_argument("--zip", required=True, help="Path to ZIP file containing DICOM folder(s).")
    parser.add_argument("--work-dir", default="./work_cases",
                        help="Working directory for intermediate files.")
    parser.add_argument("--output", default="results.json",
                        help="Output JSON file path.")
    parser.add_argument("--series-id", default=None,
                        help="Optional: exact SeriesInstanceUID to use.")
    args = parser.parse_args()

    run_pipeline(
        zip_path=args.zip,
        work_dir=args.work_dir,
        output_path=args.output,
        series_id=args.series_id,
    )


if __name__ == "__main__":
    main()
