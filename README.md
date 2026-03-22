# LUNA25 — Lung Nodule Detection & Malignancy Prediction

Pipeline tự động phát hiện nốt phổi (nodule) từ ảnh CT DICOM và dự đoán lành tính / ác tính.

## Kiến trúc

```
DICOM ZIP ──▶ NIfTI ──▶ MONAI Detection ──▶ Extract 3D Block ──▶ Predict Malignancy ──▶ JSON
```

| Module | Mô tả |
|---|---|
| `concat_3d.py` | Giải nén ZIP DICOM, chọn series CT phổi tốt nhất, convert sang NIfTI |
| `use_monai_detect.py` | Phát hiện nodule bằng MONAI RetinaNet (3D) |
| `extract_nodule.py` | Trích xuất khối ảnh 3D (64×128×128) quanh tâm nodule |
| `predictor.py` | Dự đoán ác tính / lành tính bằng ResNet3D + Tabular (tuổi, giới tính) |
| `tabular.py` | Kiến trúc model ResNet3D\_Tabular |
| **`pipeline.py`** | **Script chính** — kết nối tất cả module thành pipeline hoàn chỉnh |

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách sử dụng

### Pipeline đầy đủ (khuyến nghị)

```bash
python pipeline.py --zip path/to/dicom.zip
```

**Tham số:**

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--zip` | *(bắt buộc)* | Đường dẫn file ZIP chứa DICOM |
| `--work-dir` | `./work_cases` | Thư mục lưu file trung gian |
| `--output` | `results.json` | Đường dẫn file JSON kết quả |
| `--series-id` | `None` | Chỉ định SeriesInstanceUID cụ thể (tự động chọn nếu bỏ trống) |

### Chạy từng bước riêng lẻ

```bash
# 1. Convert DICOM → NIfTI
python concat_3d.py --zip path/to/dicom.zip

# 2. Detect nodules (cần file image.nii.gz trong thư mục hiện tại)
python use_monai_detect.py
```

## Kết quả đầu ra

File JSON chứa danh sách các nodule, mỗi nodule có dạng:

```json
{
  "seriesInstanceUID": "1.2.840.113619.2.417...",
  "probability": 0.5120,
  "predictionLabel": 1,
  "processingTimeMs": 588,
  "CoordX": 154.42,
  "CoordY": 105.35,
  "CoordZ": -163.54
}
```

| Trường | Ý nghĩa |
|---|---|
| `seriesInstanceUID` | UID của series DICOM được chọn |
| `probability` | Xác suất ác tính (0–1) |
| `predictionLabel` | `1` = Ác tính, `0` = Lành tính |
| `processingTimeMs` | Thời gian dự đoán (ms) |
| `CoordX/Y/Z` | Tọa độ thực (mm) của tâm nodule |

## Models cần có

Đặt các file sau trong thư mục gốc hoặc `assets/`:

| File | Mô tả |
|---|---|
| `assets/dt_model.ts` | MONAI RetinaNet detection model (TorchScript) |
| `best_model_by_auc.pth` | ResNet3D\_Tabular prediction model weights |
| `tabular_preprocessor_notkaggle.pkl` | Scaler cho dữ liệu tabular (Age, Gender) |

## Ghi chú

- **Tuổi** và **giới tính** được tự động trích xuất từ DICOM metadata (`PatientAge`, `PatientSex`)
- Nếu không tìm thấy thông tin bệnh nhân, giá trị mặc định: 60 tuổi, Male
- Hỗ trợ chạy trên cả **CPU** và **GPU** (tự động detect)
