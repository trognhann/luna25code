import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
import joblib
import os
import SimpleITK as sitk

from tqdm import tqdm

from tabular import ResNet3D_Tabular, normalize_ct_scan

class NodulePredictor:
    def __init__(self, model_path, preprocessor_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on: {self.device}")

        # 1. Load Preprocessor
        self.tabular_preprocessor = joblib.load(preprocessor_path)

        # 2. Load Model
        # tabular_features=2 vì chúng ta dùng Age và Gender
        self.model = ResNet3D_Tabular(tabular_features=2).to(self.device)

        # Load weights (map_location để tránh lỗi khi train GPU mà infer CPU)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.eval() # Chuyển sang chế độ đánh giá (tắt Dropout, v.v.)
        print("Model loaded successfully!")

    def predict(self, image_block, age, gender_str):
        """
        Dự đoán cho 1 mẫu duy nhất.
        - image_block: Đường dẫn đến file .npy của khối ảnh 3D
        - age: Tuổi (số)
        - gender_str: 'Male' hoặc 'Female'
        """
        # --- A. Xử lý ảnh ---
        # Normalize
        image_block = normalize_ct_scan(image_block)
        # Chuyển sang Tensor (1, 1, D, H, W) -> Batch size = 1, Channel = 1
        image_tensor = torch.tensor(image_block, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # --- B. Xử lý Tabular ---
        # Tạo DataFrame tạm để khớp định dạng với lúc train
        df_tab = pd.DataFrame({'Age_at_StudyDate': [age], 'Gender': [gender_str]})
        # Map Gender
        df_tab['Gender'] = df_tab['Gender'].map({'Male': 1, 'Female': 0})
        # Scale dữ liệu
        tab_scaled = self.tabular_preprocessor.transform(df_tab)
        # Chuyển sang Tensor
        tab_tensor = torch.tensor(tab_scaled, dtype=torch.float32).to(self.device)

        # --- C. Dự đoán ---
        with torch.no_grad():
            logit = self.model(image_tensor, tab_tensor)
            prob = torch.sigmoid(logit).item()

        prediction = 1 if prob > 0.5 else 0
        label_name = "Ác tính (Malignant)" if prediction == 1 else "Lành tính (Benign)"

        return {
            "probability": prob,
            "prediction": prediction,
            "label": label_name
        }

MODEL_PATH = 'best_model_by_auc.pth' # File model đã lưu
PREPROCESSOR_PATH = 'tabular_preprocessor_notkaggle.pkl' # File scaler đã lưu

_PREDICTOR = None

def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = NodulePredictor(MODEL_PATH, PREPROCESSOR_PATH)
    return _PREDICTOR

def predictor_main(image_block, age, gender):
    predictor = _get_predictor()

    print(f"\nĐang dự đoán cho bệnh nhân: {age} tuổi, giới tính {gender}...")
    result = predictor.predict(image_block, age, gender)

    print("--- KẾT QUẢ ---")
    print(f"Xác suất ác tính: {result['probability']:.4f}")
    print(f"Kết luận: {result['label']}")
    return {
        "probability": result["probability"],
        "predictionLabel": result["prediction"],
        "label": result["label"],
    }


if __name__ == "__main__":
    test_image_path = r'nodule.npy'
    image_block = np.load(test_image_path)
    test_age = 59
    test_gender = 'Female'
    predictor_main(image_block, test_age, test_gender)