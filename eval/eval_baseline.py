import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as skimage_ssim

from fvd import I3DExtractor, compute_fvd, extract_features
from ocr_flux_text_add_metrics import (
    TextRecognizer,
    compute_metrics,
    create_predictor,
    recognize_masked_text,
)
from style_withdataset_batch import calculate_metrics_for_images


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Sample:
    index: int
    src_video: Path
    pred_video: Path
    mask_media: Path
    src_text: str
    gt_text: str


@dataclass
class PreparedSample:
    sample: Sample
    src_frame_path: Path
    pred_frame_path: Path
    mask_frame_path: Path
    component_mask_path: Optional[Path]
    src_fg_crop_path: Optional[Path]
    pred_fg_crop_path: Optional[Path]
    src_fg_square_path: Optional[Path]
    pred_fg_square_path: Optional[Path]
    src_bg_path: Optional[Path]
    pred_bg_path: Optional[Path]


@dataclass
class FVDContext:
    model: Optional[torch.nn.Module] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline dirs with split metrics (syn/real).")
    parser.add_argument("--src_dir", type=str, default="output/vte_exp/tgt")
    parser.add_argument(
        "--src_name_suffix",
        type=str,
        default="tgt",
        help="Source video filename suffix in src_dir, e.g. 000_<suffix>.mp4",
    )
    parser.add_argument("--edited_dir", nargs="+", required=True, help="One or more edited video dirs.")
    parser.add_argument("--csv_syn", type=str, default="data/video_text_data/synbench_prompt1_first100.csv")
    parser.add_argument("--csv_real", type=str, default="data/video_text_data/realbench_prompt1_first100.csv")
    parser.add_argument("--tmp_root", type=str, default="output/vte_exp/eval_tmp")
    parser.add_argument("--result_root", type=str, default="output/vte_exp")
    parser.add_argument("--save_tag", type=str, default="v1")

    parser.add_argument("--num_syn", type=int, default=100)
    parser.add_argument("--num_real", type=int, default=100)

    parser.add_argument("--num_regions", type=int, default=1)
    parser.add_argument("--min_component_area", type=int, default=32)
    parser.add_argument(
        "--frame_number",
        type=int,
        default=20,
        help="1-based frame number used for image-level metrics (OCR/style/FID image sets).",
    )
    parser.add_argument(
        "--square_size",
        type=int,
        default=224,
        help="Side length for square-padded foreground image/video before FID/FVD.",
    )

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")

    default_rec_model = "models/SteerVTE/ocr_weights/ppv3_rec.pth"
    default_rec_dict = "models/SteerVTE/ocr_weights/ppocr_keys_v1.txt"
    parser.add_argument("--rec_model_path", type=str, default=default_rec_model)
    parser.add_argument("--rec_char_dict_path", type=str, default=default_rec_dict)
    parser.add_argument("--rec_image_shape", type=str, default="3,48,320")
    parser.add_argument("--rec_batch_num", type=int, default=64)
    parser.add_argument("--model_lang", type=str, default="ch", choices=["ch", "en"])

    parser.add_argument("--fid_batch_size", type=int, default=1)
    parser.add_argument("--fid_dims", type=int, default=2048)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", str(tag).strip())
    return cleaned if cleaned else "v1"


def method_tag(path: Path) -> str:
    def sanitize(name: str) -> str:
        tag = re.sub(r"[^a-zA-Z0-9._-]", "_", name.strip())
        return tag if tag else "method"

    parts = [p for p in path.parts if p not in ("", os.sep)]
    lower_parts = [p.lower() for p in parts]

    # Prefer the last meaningful directory under ".../vte_exp/...".
    # For example:
    # output/vte_exp/scene_text/flux2/videos -> flux2
    if "vte_exp" in lower_parts:
        idx = lower_parts.index("vte_exp")
        generic_parts = {
            "scene_text",
            "scene_text_merged",
            "outputs",
            "output",
            "tgt",
            "src",
            "results",
            "videos",
            "video",
        }
        for candidate in reversed(parts[idx + 1 :]):
            if candidate.lower() not in generic_parts:
                return sanitize(candidate)

    # Fallback: use basename, but skip generic tail names.
    generic_tail = {"outputs", "output", "tgt", "src", "results", "videos", "video"}
    candidate = path.name
    if candidate.lower() in generic_tail and path.parent.name:
        candidate = path.parent.name
    return sanitize(candidate)


def read_media_frame(path: Path, frame_index: int, as_gray: bool = False) -> Optional[np.ndarray]:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
        return cv2.imread(str(path), flag)

    if ext not in VIDEO_EXTS:
        return None

    cap = cv2.VideoCapture(str(path))
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    if (not ok or frame is None) and frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if as_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def crop_with_component_mask(image_bgr: np.ndarray, component_mask: np.ndarray) -> Optional[np.ndarray]:
    if image_bgr is None or component_mask is None:
        return None

    h, w = image_bgr.shape[:2]
    if component_mask.shape[:2] != (h, w):
        component_mask = cv2.resize(component_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    ys, xs = (component_mask > 0).nonzero()
    if len(xs) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    crop_img = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = component_mask[y1 : y2 + 1, x1 : x2 + 1]
    crop = (crop_img * (crop_mask[..., None] > 0) + 255 * (crop_mask[..., None] == 0)).astype("uint8")
    return crop


def crop_with_bbox_and_mask(
    image_bgr: np.ndarray,
    mask_bin: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    h, w = image_bgr.shape[:2]
    if mask_bin.shape[:2] != (h, w):
        mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 < x1 or y2 < y1:
        return None

    crop_img = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = mask_bin[y1 : y2 + 1, x1 : x2 + 1]
    if crop_img.size == 0 or crop_mask.size == 0:
        return None
    crop = (crop_img * (crop_mask[..., None] > 0) + 255 * (crop_mask[..., None] == 0)).astype("uint8")
    return crop


def largest_component_mask(mask_gray: np.ndarray) -> Optional[np.ndarray]:
    if mask_gray is None:
        return None

    _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (mask_bin > 0).astype("uint8")
    if binary.max() == 0:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas) + 1)
    return (labels == label).astype("uint8")


def mask_bbox(mask_bin: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = (mask_bin > 0).nonzero()
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)


def paste_center_square_and_resize(image_bgr: np.ndarray, out_size: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    top = (side - h) // 2
    left = (side - w) // 2
    square[top : top + h, left : left + w] = image_bgr
    if side != out_size:
        square = cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return square


def apply_mask_white_background(image_bgr: np.ndarray, mask_bin: np.ndarray, keep_foreground: bool) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if mask_bin.shape[:2] != (h, w):
        mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    keep = (mask_bin > 0) if keep_foreground else (mask_bin == 0)
    out = image_bgr.copy()
    out[~keep] = 255
    return out


def compute_basic_metrics(src_bgr: np.ndarray, pred_bgr: np.ndarray) -> Dict[str, float]:
    if src_bgr.shape[:2] != pred_bgr.shape[:2]:
        pred_bgr = cv2.resize(pred_bgr, (src_bgr.shape[1], src_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pred = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mse = float(np.mean((src - pred) ** 2))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))

    ssim = float(skimage_ssim(src, pred, data_range=1.0, channel_axis=-1))
    return {"ssim": ssim, "psnr": psnr, "mse": mse}


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def compute_masked_fg_bg_basic_metrics(
    src_bgr: np.ndarray,
    pred_bgr: np.ndarray,
    mask_bin: np.ndarray,
) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    if src_bgr is None or pred_bgr is None or mask_bin is None:
        return None

    h, w = src_bgr.shape[:2]
    if pred_bgr.shape[:2] != (h, w):
        pred_bgr = cv2.resize(pred_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    if mask_bin.shape[:2] != (h, w):
        mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

    fg_mask = mask_bin > 0
    bg_mask = ~fg_mask

    src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pred = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    diff_sq = (src - pred) ** 2

    def _masked_mse(mask: np.ndarray) -> Optional[float]:
        if not np.any(mask):
            return None
        return float(diff_sq[mask].mean())

    mse_fg = _masked_mse(fg_mask)
    mse_bg = _masked_mse(bg_mask)

    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    pred_gray = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    try:
        _, ssim_map = skimage_ssim(src_gray, pred_gray, data_range=1.0, full=True)
    except Exception:
        ssim_map = None

    def _masked_ssim(mask: np.ndarray) -> Optional[float]:
        if ssim_map is None or (not np.any(mask)):
            return None
        return float(ssim_map[mask].mean())

    ssim_fg = _masked_ssim(fg_mask)
    ssim_bg = _masked_ssim(bg_mask)

    return {
        "fg": {
            "ssim": ssim_fg,
            "mse": mse_fg,
            "psnr": None if mse_fg is None else _psnr_from_mse(mse_fg),
        },
        "bg": {
            "ssim": ssim_bg,
            "mse": mse_bg,
            "psnr": None if mse_bg is None else _psnr_from_mse(mse_bg),
        },
    }


def bgr_to_tensor_rgb_float(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def build_samples_for_split(
    src_dir: Path,
    edited_dir: Path,
    csv_path: Path,
    base_index: int,
    limit: int,
    src_name_suffix: str,
    src_from_csv: bool = False,
    csv_root: Path = Path("."),
    pred_name_width: int = 3,
    pred_index_base: int = 0,
    row_offset: int = 0,
    skip_missing_pred: bool = False,
) -> Tuple[List[Sample], int]:
    df = pd.read_csv(csv_path)
    selected_df = df.iloc[row_offset : row_offset + limit]
    count = len(selected_df)

    samples: List[Sample] = []
    for i, (_, row) in enumerate(selected_df.iterrows()):
        idx = base_index + i
        if src_from_csv:
            src_video = Path(str(row["video"]))
            mask_media = Path(str(row["vace_video_mask"]))
            if not src_video.is_absolute():
                src_video = csv_root / src_video
            if not mask_media.is_absolute():
                mask_media = csv_root / mask_media
        else:
            src_video = src_dir / f"{idx:03d}_{src_name_suffix}.mp4"
            mask_media = Path(str(row["vace_video_mask"]))
        if pred_name_width > 0:
            pred_name = f"{idx + pred_index_base:0{pred_name_width}d}.mp4"
        else:
            pred_name = f"{idx + pred_index_base}.mp4"
        pred_video = edited_dir / pred_name
        if skip_missing_pred and not pred_video.is_file():
            print(f"[SKIP] Missing prediction: {pred_video}")
            continue
        samples.append(
            Sample(
                index=idx,
                src_video=src_video,
                pred_video=pred_video,
                mask_media=mask_media,
                src_text=safe_text(row.get("src_text", "")),
                gt_text=safe_text(row.get("gt_text", "")),
            )
        )
    return samples, count


def prepare_samples(samples: List[Sample], tmp_dir: Path, args: argparse.Namespace) -> List[PreparedSample]:
    frames_src_dir = tmp_dir / "frames" / "src"
    frames_pred_dir = tmp_dir / "frames" / "pred"
    masks_dir = tmp_dir / "masks"
    comp_masks_dir = tmp_dir / "component_masks"
    fg_src_dir = tmp_dir / "crop_images" / "fg" / "src"
    fg_pred_dir = tmp_dir / "crop_images" / "fg" / "pred"
    bg_src_dir = tmp_dir / "crop_images" / "bg" / "src"
    bg_pred_dir = tmp_dir / "crop_images" / "bg" / "pred"

    for d in [
        frames_src_dir,
        frames_pred_dir,
        masks_dir,
        comp_masks_dir,
        fg_src_dir,
        fg_pred_dir,
        bg_src_dir,
        bg_pred_dir,
    ]:
        ensure_dir(d)

    prepared: List[PreparedSample] = []
    frame_index = max(0, int(args.frame_number) - 1)

    def load_sample_media(sample: Sample):
        return (
            sample,
            read_media_frame(sample.src_video, frame_index=frame_index, as_gray=False),
            read_media_frame(sample.pred_video, frame_index=frame_index, as_gray=False),
            read_media_frame(sample.mask_media, frame_index=frame_index, as_gray=True),
        )

    worker_count = max(1, int(getattr(args, "prepare_workers", 1)))

    def process_loaded_sample(s, src_img, pred_img, mask_img):
        stem = f"{s.index:03d}"
        src_frame_path = frames_src_dir / f"{stem}.jpg"
        pred_frame_path = frames_pred_dir / f"{stem}.jpg"
        mask_frame_path = masks_dir / f"{stem}.png"
        comp_mask_path = comp_masks_dir / f"{stem}.png"
        src_fg_crop_path = fg_src_dir / f"{stem}.png"
        pred_fg_crop_path = fg_pred_dir / f"{stem}.png"
        src_bg_path = bg_src_dir / f"{stem}.png"
        pred_bg_path = bg_pred_dir / f"{stem}.png"

        if src_img is not None:
            cv2.imwrite(str(src_frame_path), src_img)
        if pred_img is not None:
            cv2.imwrite(str(pred_frame_path), pred_img)
        if mask_img is not None:
            cv2.imwrite(str(mask_frame_path), mask_img)

        final_comp_mask = None
        final_src_fg_crop = None
        final_pred_fg_crop = None
        final_src_bg = None
        final_pred_bg = None
        if src_img is not None and pred_img is not None and mask_img is not None:
            comp = largest_component_mask(mask_img)
            if comp is not None:
                comp_u8 = (comp > 0).astype("uint8") * 255
                cv2.imwrite(str(comp_mask_path), comp_u8)
                final_comp_mask = comp_mask_path

                src_bg = apply_mask_white_background(src_img, comp, keep_foreground=False)
                pred_bg = apply_mask_white_background(pred_img, comp, keep_foreground=False)
                cv2.imwrite(str(src_bg_path), src_bg)
                cv2.imwrite(str(pred_bg_path), pred_bg)
                final_src_bg = src_bg_path
                final_pred_bg = pred_bg_path

                bbox = mask_bbox(comp)
                if bbox is not None:
                    src_fg_crop = crop_with_bbox_and_mask(src_img, comp, bbox)
                    pred_fg_crop = crop_with_bbox_and_mask(pred_img, comp, bbox)
                    if src_fg_crop is not None and pred_fg_crop is not None:
                        cv2.imwrite(str(src_fg_crop_path), src_fg_crop)
                        cv2.imwrite(str(pred_fg_crop_path), pred_fg_crop)
                        final_src_fg_crop = src_fg_crop_path
                        final_pred_fg_crop = pred_fg_crop_path


        return PreparedSample(
                sample=s,
                src_frame_path=src_frame_path,
                pred_frame_path=pred_frame_path,
                mask_frame_path=mask_frame_path,
                component_mask_path=final_comp_mask,
                src_fg_crop_path=final_src_fg_crop,
                pred_fg_crop_path=final_pred_fg_crop,
                src_fg_square_path=None,
                pred_fg_square_path=None,
                src_bg_path=final_src_bg,
                pred_bg_path=final_pred_bg,
        )

    for batch_start in range(0, len(samples), worker_count):
        batch = samples[batch_start : batch_start + worker_count]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            loaded_samples = executor.map(load_sample_media, batch)
            for loaded_sample in loaded_samples:
                prepared.append(process_loaded_sample(*loaded_sample))

    return prepared


def build_ocr_recognizer(args: argparse.Namespace, device: torch.device) -> TextRecognizer:
    predictor = create_predictor(args.rec_model_path, model_lang=args.model_lang).to(device).eval()

    use_fp16 = bool(args.use_fp16 and device.type == "cuda")
    if use_fp16:
        predictor = predictor.half()

    return TextRecognizer(
        rec_image_shape=args.rec_image_shape,
        rec_batch_num=args.rec_batch_num,
        rec_char_dict_path=args.rec_char_dict_path,
        predictor=predictor,
        device=device,
        use_fp16=use_fp16,
    )


def evaluate_ocr(prepared: List[PreparedSample], args: argparse.Namespace, recognizer: TextRecognizer) -> Dict:
    per_sample = []
    valid = 0
    sum_metrics = {
        "mean_sen_acc": 0.0,
        "mean_ned": 0.0,
        "mean_precision": 0.0,
        "mean_recall": 0.0,
        "mean_f_score": 0.0,
    }

    for item in prepared:
        entry = {
            "index": item.sample.index,
            "pred_video": str(item.sample.pred_video),
            "pred_frame": str(item.pred_frame_path),
            "mask_frame": str(item.mask_frame_path),
            "src_text": item.sample.src_text,
            "gt_text": item.sample.gt_text,
            "pred_rec_text": "",
            "sen_acc": 0.0,
            "ned": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f_score": 0.0,
            "status": "missing",
        }

        pred_img = cv2.imread(str(item.pred_frame_path), cv2.IMREAD_COLOR)
        mask_img = cv2.imread(str(item.mask_frame_path), cv2.IMREAD_GRAYSCALE)
        if pred_img is None or mask_img is None:
            per_sample.append(entry)
            continue

        if mask_img.shape[:2] != pred_img.shape[:2]:
            mask_img = cv2.resize(mask_img, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_rec_text = recognize_masked_text(
            pred_img,
            mask_img,
            recognizer,
            num_regions=args.num_regions,
            min_component_area=args.min_component_area,
        )
        sen_acc, ned, precision, recall, f_score = compute_metrics(pred_rec_text, item.sample.gt_text)

        entry.update(
            {
                "pred_rec_text": pred_rec_text,
                "sen_acc": float(sen_acc),
                "ned": float(ned),
                "precision": float(precision),
                "recall": float(recall),
                "f_score": float(f_score),
                "status": "ok",
            }
        )

        valid += 1
        sum_metrics["mean_sen_acc"] += entry["sen_acc"]
        sum_metrics["mean_ned"] += entry["ned"]
        sum_metrics["mean_precision"] += entry["precision"]
        sum_metrics["mean_recall"] += entry["recall"]
        sum_metrics["mean_f_score"] += entry["f_score"]
        per_sample.append(entry)

    if valid > 0:
        for k in list(sum_metrics.keys()):
            sum_metrics[k] /= valid
    else:
        sum_metrics = {
            "mean_sen_acc": 0.0,
            "mean_ned": 1.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f_score": 0.0,
        }

    return {
        "total_rows": len(prepared),
        "valid_rows": valid,
        "missing_rows": len(prepared) - valid,
        **sum_metrics,
        "per_sample": per_sample,
    }


def evaluate_style(prepared: List[PreparedSample], args: argparse.Namespace, device: torch.device, tmp_dir: Path) -> Dict:
    per_sample = []
    valid = 0
    sum_fg_style = {
        "ssim": 0.0,
        "psnr": 0.0,
        "mse": 0.0,
        "s_color": 0.0,
        "s_font": 0.0,
        "s_bg": 0.0,
        "tas_score": 0.0,
    }
    sum_bg_basic = {
        "ssim": 0.0,
        "psnr": 0.0,
        "mse": 0.0,
    }
    valid_bg = 0

    for item in prepared:
        entry = {
            "index": item.sample.index,
            "src_fg_crop": str(item.src_fg_crop_path) if item.src_fg_crop_path else "",
            "pred_fg_crop": str(item.pred_fg_crop_path) if item.pred_fg_crop_path else "",
            "src_frame": str(item.src_frame_path),
            "pred_frame": str(item.pred_frame_path),
            "component_mask": str(item.component_mask_path) if item.component_mask_path else "",
            "src_bg": str(item.src_bg_path) if item.src_bg_path else "",
            "pred_bg": str(item.pred_bg_path) if item.pred_bg_path else "",
            "status": "missing",
        }

        if item.src_fg_crop_path is None or item.pred_fg_crop_path is None or item.component_mask_path is None:
            per_sample.append(entry)
            continue

        src_frame = cv2.imread(str(item.src_frame_path), cv2.IMREAD_COLOR)
        pred_frame = cv2.imread(str(item.pred_frame_path), cv2.IMREAD_COLOR)
        comp_mask = cv2.imread(str(item.component_mask_path), cv2.IMREAD_GRAYSCALE)
        if src_frame is None or pred_frame is None or comp_mask is None:
            per_sample.append(entry)
            continue

        masked_basic = compute_masked_fg_bg_basic_metrics(src_frame, pred_frame, comp_mask)
        if masked_basic is None:
            per_sample.append(entry)
            continue
        fg_basic = masked_basic["fg"]
        bg_basic = masked_basic["bg"]
        if fg_basic["ssim"] is None or fg_basic["psnr"] is None or fg_basic["mse"] is None:
            per_sample.append(entry)
            continue

        src_fg_crop = cv2.imread(str(item.src_fg_crop_path), cv2.IMREAD_COLOR)
        pred_fg_crop = cv2.imread(str(item.pred_fg_crop_path), cv2.IMREAD_COLOR)
        if src_fg_crop is None or pred_fg_crop is None:
            per_sample.append(entry)
            continue

        src_h, src_w = src_fg_crop.shape[:2]
        pred_h, pred_w = pred_fg_crop.shape[:2]
        if (src_h, src_w) != (pred_h, pred_w):
            # Some methods output frames at a different resolution from src.
            # Align pred crop to src crop size before pairwise pixel metrics.
            pred_fg_crop = cv2.resize(pred_fg_crop, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
            entry["resized_for_metric"] = True
            entry["src_shape"] = [int(src_h), int(src_w)]
            entry["pred_shape_raw"] = [int(pred_h), int(pred_w)]

        try:
            pred_tensor = bgr_to_tensor_rgb_float(pred_fg_crop)
            src_tensor = bgr_to_tensor_rgb_float(src_fg_crop)
            metrics = calculate_metrics_for_images(
                pred_tensor,
                src_tensor,
                device=device,
                img_path=str(item.pred_fg_crop_path),
                gt_path=str(item.src_fg_crop_path),
            )
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
            per_sample.append(entry)
            continue

        entry.update(
            {
                "status": "ok",
                "fg": {
                    "ssim": float(fg_basic["ssim"]),
                    "psnr": float(fg_basic["psnr"]),
                    "mse": float(fg_basic["mse"]),
                    "s_color": float(metrics["s_color"]),
                    "s_font": float(metrics["s_font"]),
                    "s_bg": float(metrics["s_bg"]),
                    "tas_score": float(metrics["tas_score"]),
                },
            }
        )
        if bg_basic["ssim"] is not None and bg_basic["psnr"] is not None and bg_basic["mse"] is not None:
            entry["bg"] = {
                "ssim": float(bg_basic["ssim"]),
                "psnr": float(bg_basic["psnr"]),
                "mse": float(bg_basic["mse"]),
            }
            valid_bg += 1
            for k in sum_bg_basic.keys():
                sum_bg_basic[k] += entry["bg"][k]
        else:
            entry["bg"] = None

        for k in sum_fg_style.keys():
            sum_fg_style[k] += entry["fg"][k]
        valid += 1
        per_sample.append(entry)

    avg_fg = {k: (sum_fg_style[k] / valid if valid > 0 else 0.0) for k in sum_fg_style.keys()}
    avg_bg = {k: (sum_bg_basic[k] / valid_bg if valid_bg > 0 else 0.0) for k in sum_bg_basic.keys()}

    return {
        "total_rows": len(prepared),
        "valid_rows": valid,
        "valid_bg_rows": valid_bg,
        "missing_rows": len(prepared) - valid,
        "fg": avg_fg,
        "bg": avg_bg,
        "per_sample": per_sample,
    }


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    src = src.resolve()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def calc_fid_for_dirs(
    src_dir: Path,
    pred_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    pair_tmp_dir: Optional[Path] = None,
) -> Dict:
    if not src_dir.exists() or not pred_dir.exists():
        return {"value": None, "error": "dir not found", "num_pairs": 0}

    src_files = {p.name: p for p in src_dir.iterdir() if p.is_file()}
    pred_files = {p.name: p for p in pred_dir.iterdir() if p.is_file()}
    common_names = sorted(set(src_files.keys()).intersection(set(pred_files.keys())))
    pair_count = len(common_names)
    if pair_count == 0:
        return {"value": None, "error": "no valid pairs", "num_pairs": 0}

    run_src_dir = src_dir
    run_pred_dir = pred_dir
    if pair_tmp_dir is not None:
        if pair_tmp_dir.exists():
            shutil.rmtree(pair_tmp_dir)
        run_src_dir = pair_tmp_dir / "src"
        run_pred_dir = pair_tmp_dir / "pred"
        ensure_dir(run_src_dir)
        ensure_dir(run_pred_dir)
        for name in common_names:
            link_or_copy(src_files[name], run_src_dir / name)
            link_or_copy(pred_files[name], run_pred_dir / name)

    try:
        value = float(
            fid_score.calculate_fid_given_paths(
                [str(run_src_dir), str(run_pred_dir)],
                args.fid_batch_size,
                device,
                args.fid_dims,
            )
        )
        return {"value": value, "error": None, "num_pairs": pair_count}
    except Exception as e:
        return {"value": None, "error": str(e), "num_pairs": pair_count}


def build_fvd_pair_dirs_raw(samples: List[Sample], tmp_dir: Path) -> Tuple[Path, Path, int]:
    src_dir = tmp_dir / "fvd" / "raw" / "src"
    pred_dir = tmp_dir / "fvd" / "raw" / "pred"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    ensure_dir(src_dir)
    ensure_dir(pred_dir)

    pair_count = 0
    for s in samples:
        if not s.src_video.exists() or not s.pred_video.exists():
            continue
        src_ext = s.src_video.suffix.lower() or ".mp4"
        pred_ext = s.pred_video.suffix.lower() or ".mp4"
        src_out = src_dir / f"{s.index:03d}{src_ext}"
        pred_out = pred_dir / f"{s.index:03d}{pred_ext}"
        link_or_copy(s.src_video, src_out)
        link_or_copy(s.pred_video, pred_out)
        pair_count += 1

    return src_dir, pred_dir, pair_count


def evaluate_fvd_dir_pair(src_dir: Path, pred_dir: Path, pair_count: int, fvd_ctx: FVDContext) -> Dict:
    if pair_count == 0:
        return {"value": None, "error": "no valid video pairs", "num_pairs": 0}

    try:
        if fvd_ctx.model is None:
            fvd_ctx.model = I3DExtractor()

        real_feats = extract_features(str(src_dir), fvd_ctx.model)
        fake_feats = extract_features(str(pred_dir), fvd_ctx.model)
        value = float(compute_fvd(real_feats, fake_feats))
        return {"value": value, "error": None, "num_pairs": pair_count}
    except Exception as e:
        return {"value": None, "error": str(e), "num_pairs": pair_count}


def build_fvd_pair_dirs_processed(
    prepared: List[PreparedSample],
    tmp_dir: Path,
    kind: str,
    mode: str,
    out_size: int,
) -> Tuple[Path, Path, int]:
    src_dir = tmp_dir / "fvd" / kind / "src"
    pred_dir = tmp_dir / "fvd" / kind / "pred"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    ensure_dir(src_dir)
    ensure_dir(pred_dir)

    pair_count = 0
    for item in prepared:
        s = item.sample
        if not s.src_video.exists() or not s.pred_video.exists():
            continue
        if item.component_mask_path is None:
            continue
        mask = cv2.imread(str(item.component_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_bin = (mask > 0).astype("uint8")
        bbox = mask_bbox(mask_bin)

        src_cap = cv2.VideoCapture(str(s.src_video))
        pred_cap = cv2.VideoCapture(str(s.pred_video))
        src_ext = s.src_video.suffix.lower() or ".mp4"
        pred_ext = s.pred_video.suffix.lower() or ".mp4"
        src_out = src_dir / f"{s.index:03d}{src_ext}"
        pred_out = pred_dir / f"{s.index:03d}{pred_ext}"

        src_fps = src_cap.get(cv2.CAP_PROP_FPS)
        pred_fps = pred_cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 1e-6:
            src_fps = 8.0
        if pred_fps <= 1e-6:
            pred_fps = 8.0

        if mode == "square":
            out_w, out_h = out_size, out_size
        elif mode == "crop":
            if bbox is None:
                src_cap.release()
                pred_cap.release()
                continue
            x1, y1, x2, y2 = bbox
            out_w, out_h = int(x2 - x1 + 1), int(y2 - y1 + 1)
            if out_w <= 0 or out_h <= 0:
                src_cap.release()
                pred_cap.release()
                continue
        else:
            out_w, out_h = mask.shape[1], mask.shape[0]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        src_writer = cv2.VideoWriter(str(src_out), fourcc, float(src_fps), (int(out_w), int(out_h)))
        pred_writer = cv2.VideoWriter(str(pred_out), fourcc, float(pred_fps), (int(out_w), int(out_h)))
        if not src_writer.isOpened() or not pred_writer.isOpened():
            src_cap.release()
            pred_cap.release()
            src_writer.release()
            pred_writer.release()
            continue

        wrote_src = 0
        wrote_pred = 0
        while True:
            ok, frame = src_cap.read()
            if not ok or frame is None:
                break
            proc = frame
            if proc.shape[:2] != mask.shape[:2]:
                proc = cv2.resize(proc, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            proc = apply_mask_white_background(proc, mask_bin, keep_foreground=True)
            if mode == "square":
                if bbox is None:
                    proc = np.full((out_size, out_size, 3), 255, dtype=np.uint8)
                else:
                    crop = crop_with_bbox_and_mask(proc, mask_bin, bbox)
                    if crop is None:
                        proc = np.full((out_size, out_size, 3), 255, dtype=np.uint8)
                    else:
                        proc = paste_center_square_and_resize(crop, out_size=out_size)
            elif mode == "crop":
                crop = crop_with_bbox_and_mask(proc, mask_bin, bbox) if bbox is not None else None
                if crop is None:
                    proc = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
                else:
                    proc = crop
            elif proc.shape[1] != out_w or proc.shape[0] != out_h:
                proc = cv2.resize(proc, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            src_writer.write(proc)
            wrote_src += 1

        while True:
            ok, frame = pred_cap.read()
            if not ok or frame is None:
                break
            proc = frame
            if proc.shape[:2] != mask.shape[:2]:
                proc = cv2.resize(proc, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            proc = apply_mask_white_background(proc, mask_bin, keep_foreground=True)
            if mode == "square":
                if bbox is None:
                    proc = np.full((out_size, out_size, 3), 255, dtype=np.uint8)
                else:
                    crop = crop_with_bbox_and_mask(proc, mask_bin, bbox)
                    if crop is None:
                        proc = np.full((out_size, out_size, 3), 255, dtype=np.uint8)
                    else:
                        proc = paste_center_square_and_resize(crop, out_size=out_size)
            elif mode == "crop":
                crop = crop_with_bbox_and_mask(proc, mask_bin, bbox) if bbox is not None else None
                if crop is None:
                    proc = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
                else:
                    proc = crop
            elif proc.shape[1] != out_w or proc.shape[0] != out_h:
                proc = cv2.resize(proc, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            pred_writer.write(proc)
            wrote_pred += 1

        src_cap.release()
        pred_cap.release()
        src_writer.release()
        pred_writer.release()

        if wrote_src > 0 and wrote_pred > 0:
            pair_count += 1
        else:
            if src_out.exists():
                src_out.unlink()
            if pred_out.exists():
                pred_out.unlink()

    return src_dir, pred_dir, pair_count


def evaluate_one_split(
    split_name: str,
    src_dir: Path,
    edited_dir: Path,
    csv_path: Path,
    base_index: int,
    limit: int,
    args: argparse.Namespace,
    device: torch.device,
    recognizer: TextRecognizer,
    method_tmp_dir: Path,
    fvd_ctx: FVDContext,
    src_from_csv: bool = False,
    csv_root: Path = Path("."),
    pred_name_width: int = 3,
    pred_index_base: int = 0,
    row_offset: int = 0,
    skip_missing_pred: bool = False,
) -> Dict:
    samples, actual_count = build_samples_for_split(
        src_dir=src_dir,
        edited_dir=edited_dir,
        csv_path=csv_path,
        base_index=base_index,
        limit=limit,
        src_name_suffix=args.src_name_suffix,
        src_from_csv=src_from_csv,
        csv_root=csv_root,
        pred_name_width=pred_name_width,
        pred_index_base=pred_index_base,
        row_offset=row_offset,
        skip_missing_pred=skip_missing_pred,
    )

    split_tmp_dir = method_tmp_dir / split_name
    if split_tmp_dir.exists():
        shutil.rmtree(split_tmp_dir)
    ensure_dir(split_tmp_dir)

    prepared = prepare_samples(samples, split_tmp_dir, args)
    ocr_metrics = evaluate_ocr(prepared, args, recognizer)
    style_metrics = evaluate_style(prepared, args, device, split_tmp_dir)
    fid_metrics = {
        "fg_crop": calc_fid_for_dirs(
            src_dir=split_tmp_dir / "crop_images" / "fg" / "src",
            pred_dir=split_tmp_dir / "crop_images" / "fg" / "pred",
            args=args,
            device=device,
            pair_tmp_dir=split_tmp_dir / "fid" / "fg_crop_pairs",
        )
    }

    fg_src_dir, fg_pred_dir, fg_pairs = build_fvd_pair_dirs_processed(
        prepared=prepared,
        tmp_dir=split_tmp_dir,
        kind="fg_crop",
        mode="crop",
        out_size=args.square_size,
    )

    fvd_metrics = {"fg_crop": evaluate_fvd_dir_pair(fg_src_dir, fg_pred_dir, fg_pairs, fvd_ctx)}

    return {
        "meta": {
            "split": split_name,
            "csv_path": str(csv_path),
            "base_index": base_index,
            "requested_count": limit,
            "actual_count": actual_count,
            "tmp_dir": str(split_tmp_dir),
            "frame_number": int(args.frame_number),
            "square_size": int(args.square_size),
        },
        "ocr": ocr_metrics,
        "style": style_metrics,
        "fid": fid_metrics,
        "fvd": fvd_metrics,
    }


def evaluate_one_method(
    edited_dir: Path,
    src_dir: Path,
    csv_syn: Path,
    csv_real: Path,
    args: argparse.Namespace,
    device: torch.device,
    recognizer: TextRecognizer,
    tmp_root: Path,
    fvd_ctx: FVDContext,
) -> Dict:
    method_tmp_dir = tmp_root / method_tag(edited_dir)
    if method_tmp_dir.exists():
        shutil.rmtree(method_tmp_dir)
    ensure_dir(method_tmp_dir)

    print(f"\n[METHOD] edited_dir={edited_dir}")

    syn_result = evaluate_one_split(
        split_name="syn",
        src_dir=src_dir,
        edited_dir=edited_dir,
        csv_path=csv_syn,
        base_index=0,
        limit=args.num_syn,
        args=args,
        device=device,
        recognizer=recognizer,
        method_tmp_dir=method_tmp_dir,
        fvd_ctx=fvd_ctx,
    )
    print(
        "[SYN] OCR mean_sen_acc={:.4f}, Style fg_tas_score={:.4f}, FVD(fg_crop)={}".format(
            syn_result["ocr"]["mean_sen_acc"],
            syn_result["style"]["fg"]["tas_score"],
            "nan" if syn_result["fvd"]["fg_crop"]["value"] is None else f"{syn_result['fvd']['fg_crop']['value']:.4f}",
        )
    )

    real_result = evaluate_one_split(
        split_name="real",
        src_dir=src_dir,
        edited_dir=edited_dir,
        csv_path=csv_real,
        base_index=100,
        limit=args.num_real,
        args=args,
        device=device,
        recognizer=recognizer,
        method_tmp_dir=method_tmp_dir,
        fvd_ctx=fvd_ctx,
    )
    print(
        "[REAL] OCR mean_sen_acc={:.4f}, Style fg_tas_score={:.4f}, FVD(fg_crop)={}".format(
            real_result["ocr"]["mean_sen_acc"],
            real_result["style"]["fg"]["tas_score"],
            "nan" if real_result["fvd"]["fg_crop"]["value"] is None else f"{real_result['fvd']['fg_crop']['value']:.4f}",
        )
    )

    return {
        "meta": {
            "src_dir": str(src_dir),
            "edited_dir": str(edited_dir),
            "method_name": method_tag(edited_dir),
            "tmp_dir": str(method_tmp_dir),
            "device": str(device),
            "csv_syn": str(csv_syn),
            "csv_real": str(csv_real),
        },
        "syn": syn_result,
        "real": real_result,
    }


def main() -> None:
    args = parse_args()

    src_dir = Path(args.src_dir)
    csv_syn = Path(args.csv_syn)
    csv_real = Path(args.csv_real)
    tmp_root = Path(args.tmp_root)
    result_root = Path(args.result_root)
    save_tag = sanitize_tag(args.save_tag)
    result_dir = result_root / save_tag

    if not src_dir.exists():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")
    if not csv_syn.exists():
        raise FileNotFoundError(f"csv_syn not found: {csv_syn}")
    if not csv_real.exists():
        raise FileNotFoundError(f"csv_real not found: {csv_real}")

    edited_dirs: List[Path] = []
    for d in args.edited_dir:
        p = Path(d)
        if not p.exists():
            print(f"[WARN] skip missing edited_dir: {p}")
            continue
        edited_dirs.append(p)

    if len(edited_dirs) == 0:
        raise ValueError("No valid edited_dir to evaluate.")

    ensure_dir(tmp_root)
    ensure_dir(result_dir)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    recognizer = build_ocr_recognizer(args, device)
    fvd_ctx = FVDContext(model=None)

    print("========== Baseline Eval ==========")
    print(f"src_dir: {src_dir}")
    print(f"csv_syn: {csv_syn}")
    print(f"csv_real:{csv_real}")
    print(f"tmp_root:{tmp_root}")
    print(f"result_dir:{result_dir}")
    print(f"frame_number:{args.frame_number}")
    print(f"square_size:{args.square_size}")
    print(f"device:  {device}")
    print(f"methods: {len(edited_dirs)}")
    print("===================================")

    for edited_dir in edited_dirs:
        result = evaluate_one_method(
            edited_dir=edited_dir,
            src_dir=src_dir,
            csv_syn=csv_syn,
            csv_real=csv_real,
            args=args,
            device=device,
            recognizer=recognizer,
            tmp_root=tmp_root,
            fvd_ctx=fvd_ctx,
        )

        method_name = method_tag(edited_dir)
        output_json = result_dir / f"{method_name}.json"
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved eval json: {output_json}")


if __name__ == "__main__":
    main()
