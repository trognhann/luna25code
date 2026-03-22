import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

from tqdm import tqdm

class ResNet3D_Tabular(nn.Module):
    def __init__(self, tabular_features):
        super(ResNet3D_Tabular, self).__init__()
        # 3D CNN backbone
        self.resnet18_3d = models.video.r3d_18(weights=None)
        # Sửa Conv đầu vào cho 1 channel
        self.resnet18_3d.stem[0] = nn.Conv3d(
            1,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )

        num_ftrs_cnn = self.resnet18_3d.fc.in_features
        self.resnet18_3d.fc = nn.Identity()

        # Nhánh Tabular
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Kết hợp
        self.combined_fc = nn.Sequential(
            nn.Linear(num_ftrs_cnn + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x_3d, x_tab):
        x_3d = self.resnet18_3d(x_3d)
        x_tab = self.tabular_fc(x_tab)
        x_combined = torch.cat((x_3d, x_tab), dim=1)
        return self.combined_fc(x_combined)

# ==========================================
# 2. CÁC HÀM TIỀN XỬ LÝ (PREPROCESSING)
# ==========================================
def normalize_ct_scan(image_block):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image_block = np.clip(image_block, MIN_BOUND, MAX_BOUND)
    image_block = (image_block - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return image_block.astype(np.float32)