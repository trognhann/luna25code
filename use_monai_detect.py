import json
import numpy as np
import torch

from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    DeleteItemsd,
)
from monai.apps.detection.transforms.dictionary import (
    ClipBoxToImaged,
    AffineBoxToWorldCoordinated,
    ConvertBoxModed,
)
from monai.data import Dataset, DataLoader
from monai.data.utils import no_collation

from predictor import predictor_main

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===== Paths =====
# base_dir = Path("/content")
nifti_path = "image.nii.gz"
model_path = r"assets\dt_model.ts"
output_json = "detect_result.json"

# if not nifti_path.exists():
#     raise FileNotFoundError(f"NIfTI not found: {nifti_path}")
# if not model_path.exists():
#     raise FileNotFoundError(f"Model not found: {model_path}")

# ===== Detector settings =====
spatial_dims = 3
num_classes = 1
size_divisible = [16, 16, 8]

# Giảm patch size để đỡ OOM
infer_patch_size = [192, 192, 96]
sw_batch_size = 1
overlap = 0.25
mode = "constant"

box_key = "box"
label_key = "label"

score_thresh = 0.02
topk_candidates_per_level = 300
nms_thresh = 0.22
detections_per_img = 100

# spacing đang bám theo dữ liệu bạn vừa convert
pixdim = [0.7480469942092896, 0.7480469942092896, 1.25]

# threshold cuối để giữ candidate
score_keep = 0.30


def build_preprocess(image_key="image"):
    return Compose([
        LoadImaged(keys=[image_key], reader="NibabelReader"),
        EnsureChannelFirstd(keys=[image_key]),
        Orientationd(keys=[image_key], axcodes="RAS"),
        Spacingd(
            keys=[image_key],
            pixdim=pixdim,
            mode="bilinear",
            padding_mode="border",
        ),
        ScaleIntensityRanged(
            keys=[image_key],
            a_min=-1024.0,
            a_max=300.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=[image_key]),
    ])


def build_postprocess(image_key="image", affine_lps_to_ras=True):
    return Compose([
        ClipBoxToImaged(
            box_keys="box",
            label_keys="label",
            box_ref_image_keys=image_key,
            remove_empty=True,
        ),
        AffineBoxToWorldCoordinated(
            box_keys="box",
            box_ref_image_keys=image_key,
            affine_lps_to_ras=affine_lps_to_ras,
        ),
        ConvertBoxModed(
            box_keys="box",
            src_mode="xyzxyz",
            dst_mode="cccwhd",
        ),
        DeleteItemsd(keys=[image_key]),
    ])


def build_detector(model_path, device):
    print("Loading model from:", model_path)
    network = torch.jit.load(str(model_path), map_location=device)

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[1, 2, 4],
        base_anchor_shapes=[
            [6, 8, 4],
            [8, 6, 5],
            [10, 10, 6],
        ],
    )

    detector = RetinaNetDetector(
        network=network,
        anchor_generator=anchor_generator,
        spatial_dims=spatial_dims,
        num_classes=num_classes,
        size_divisible=size_divisible,
    )
    detector.set_target_keys(box_key=box_key, label_key=label_key)
    detector.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=topk_candidates_per_level,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
    )
    detector.set_sliding_window_inferer(
        roi_size=infer_patch_size,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        mode=mode,
        device=device,
    )
    detector.eval()
    return detector


def run_detection(nifti_path_arg=None, output_json_arg=None):
    _nifti = nifti_path_arg or nifti_path
    _output = output_json_arg or output_json

    print("Preparing dataset...")
    preprocess = build_preprocess()
    postprocess = build_postprocess()
    detector = build_detector(model_path, device)

    data = [{"image": str(_nifti), "nifti_path": str(_nifti)}]
    ds = Dataset(data=data, transform=preprocess)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=no_collation,
    )

    all_results = []

    for item in dl:
        item = item[0]
        image_4d = item["image"].to(device)
        current_nifti = item["nifti_path"]

        image_for_detector = image_4d.unsqueeze(0) if image_4d.dim() == 4 else image_4d
        print("Input tensor shape:", tuple(image_for_detector.shape))

        if device == "cuda":
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                out = detector(image_for_detector, use_inferer=True)
        else:
            with torch.no_grad():
                out = detector(image_for_detector, use_inferer=True)

        out0 = out[0]
        boxes = out0["box"] if "box" in out0 else out0.get("boxes")
        labels = out0["label"] if "label" in out0 else out0.get("labels")
        scores = out0["label_scores"] if "label_scores" in out0 else out0.get("scores")

        boxes = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
        labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)

        print("Raw detections:", len(scores))

        post_out = postprocess({
            "box": boxes,
            "label": labels,
            "label_scores": scores,
            "image": image_4d,
        })

        if len(post_out["label_scores"]) > 0:
            keep = np.asarray(post_out["label_scores"]) >= float(score_keep)
            filtered_boxes = np.asarray(post_out["box"])[keep]
            filtered_scores = np.asarray(post_out["label_scores"])[keep]
            filtered_labels = np.asarray(post_out["label"])[keep]

            print("Filtered detections:", len(filtered_scores))

            for i, (box, score, label) in enumerate(
                zip(filtered_boxes, filtered_scores, filtered_labels), start=1
            ):
                cx, cy, cz, w, h, d = box.tolist()
                all_results.append({
                    "nodule_id": i,
                    "nifti_path": current_nifti,
                    "center_world": [cx, cy, cz],
                    "size_world": [w, h, d],
                    "score": float(score),
                    "label": int(label),
                })
        else:
            print("No detections after postprocess.")

    with open(_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {_output}")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))
    return all_results