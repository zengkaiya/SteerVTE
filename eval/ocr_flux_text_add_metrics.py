import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Levenshtein import distance
from easydict import EasyDict as edict
from tqdm import tqdm

from diffsynth.models.ocr_recog.RecModel import RecModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR results with Flux text recognizer.")
    parser.add_argument("--csv_path", type=str, default=None, help="CSV file path with gt_text and video metadata.")
    parser.add_argument("--infer_root", type=str, default=None, help="Visualization output root path.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save per-sample OCR results JSON.")
    parser.add_argument(
        "--job",
        nargs=3,
        action="append",
        metavar=("INFER_ROOT", "CSV_PATH", "OUTPUT_JSON"),
        help="Batch job triple. Repeat this arg to evaluate multiple jobs in one process.",
    )
    parser.add_argument(
        "--job_with_range",
        nargs=5,
        action="append",
        metavar=("INFER_ROOT", "CSV_PATH", "OUTPUT_JSON", "START_INDEX", "END_INDEX"),
        help="Batch job with per-job index range. Repeat this arg to evaluate multiple jobs in one process.",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Start row index (inclusive).")
    parser.add_argument("--end_index", type=int, default=-1, help="End row index (exclusive), -1 means full CSV.")
    parser.add_argument("--pred_image_name", type=str, default="infer_video.jpg", help="Pred first-frame image name.")
    parser.add_argument("--gt_image_name", type=str, default="video.jpg", help="GT first-frame image name.")
    parser.add_argument("--mask_image_name", type=str, default="vace_video_mask.jpg", help="Edit mask image name.")
    parser.add_argument("--num_regions", type=int, default=2, help="How many largest mask regions to OCR.")
    parser.add_argument("--min_component_area", type=int, default=32, help="Min pixel area for a mask component.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for OCR model inference.")

    default_rec_model = PROJECT_ROOT / "models/SteerVTE/ocr_weights/ppv3_rec.pth"
    default_rec_dict = PROJECT_ROOT / "models/SteerVTE/ocr_weights/ppocr_keys_v1.txt"
    parser.add_argument("--rec_model_path", type=str, default=str(default_rec_model), help="Path to OCR rec model.")
    parser.add_argument("--rec_char_dict_path", type=str, default=str(default_rec_dict), help="Path to OCR char dict.")
    parser.add_argument("--rec_image_shape", type=str, default="3,48,320", help="OCR image shape C,H,W.")
    parser.add_argument("--rec_batch_num", type=int, default=64, help="OCR batch size.")
    parser.add_argument("--model_lang", type=str, default="ch", choices=["ch", "en"], help="OCR model language.")
    parser.add_argument("--use_fp16", action="store_true", help="Use fp16 for OCR inference.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return "".join(str(text).lower().split())


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def build_relative_video_path_candidates(video_path: str, csv_path: str) -> List[str]:
    path = str(video_path).replace("\\", "/")
    path = path.replace("/clip1", "").replace("/clip2", "")
    path = os.path.splitext(path)[0]
    candidates: List[str] = []
    seen = set()

    def add_candidate(candidate: str):
        candidate = candidate.lstrip("/")
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    dataset_name = Path(csv_path).stem
    marker = f"{dataset_name}/"
    if marker in path:
        add_candidate(path.split(marker, 1)[1])

    if "/data/video_text_data/" in path:
        sub_path = path.split("/data/video_text_data/", 1)[1]
        parts = sub_path.split("/", 1)
        if len(parts) == 2:
            add_candidate(parts[1])
        add_candidate(sub_path)

    if "/data/image_text_data/" in path:
        sub_path = path.split("/data/image_text_data/", 1)[1]
        parts = sub_path.split("/")
        for anchor in ("train_imgs", "val_imgs", "test_imgs", "images", "imgs"):
            if anchor in parts:
                anchor_idx = parts.index(anchor)
                if anchor_idx + 2 < len(parts):
                    add_candidate("/".join(parts[anchor_idx + 2 :]))
                if anchor_idx + 1 < len(parts):
                    add_candidate("/".join(parts[anchor_idx + 1 :]))
        if len(parts) > 1:
            add_candidate("/".join(parts[1:]))
        add_candidate(sub_path)

    add_candidate(path)
    return candidates


def resolve_sample_dir(infer_root: Path, row_video: str, csv_path: str, args: argparse.Namespace):
    candidates = build_relative_video_path_candidates(row_video, csv_path)
    if not candidates:
        candidates = [""]

    first_existing = None
    for rel in candidates:
        sample_dir = infer_root / rel
        infer_img_path = sample_dir / args.pred_image_name
        gt_img_path = sample_dir / args.gt_image_name
        mask_img_path = sample_dir / args.mask_image_name
        if infer_img_path.exists() or gt_img_path.exists() or mask_img_path.exists():
            return rel, sample_dir
        if first_existing is None and sample_dir.exists():
            first_existing = (rel, sample_dir)

    if first_existing is not None:
        return first_existing

    rel = candidates[0]
    return rel, infer_root / rel


def resolve_named_image(sample_dir: Path, preferred_name: str) -> Path:
    base, ext = os.path.splitext(preferred_name)
    ext = ext.lower()
    fallback_exts = [".jpg", ".png", ".jpeg", ".bmp", ".webp"]
    candidates = [preferred_name]
    for e in fallback_exts:
        name = f"{base}{e}"
        if name not in candidates:
            candidates.append(name)
    if ext and ext not in fallback_exts:
        name = f"{base}{ext}"
        if name not in candidates:
            candidates.insert(0, name)

    for name in candidates:
        path = sample_dir / name
        if path.exists():
            return path

    return sample_dir / preferred_name


def extract_region_masks(mask_gray, num_regions: int, min_component_area: int) -> List[np.ndarray]:
    if mask_gray is None:
        return []

    binary = (mask_gray > 10).astype("uint8")
    if binary.max() == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        components.append((area, x, y, label))

    if not components:
        return [binary]

    components.sort(key=lambda item: item[0], reverse=True)
    components = components[: max(1, num_regions)]
    components.sort(key=lambda item: (item[1], item[2]))
    return [(labels == label).astype("uint8") for _, _, _, label in components]


def crop_with_component_mask(image_bgr, component_mask):
    ys, xs = (component_mask > 0).nonzero()
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop_img = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
    crop_mask = component_mask[y1 : y2 + 1, x1 : x2 + 1]
    crop = (crop_img * (crop_mask[..., None] > 0) + 255 * (crop_mask[..., None] == 0)).astype("uint8")
    return crop


def create_predictor(model_path: str, model_lang: str = "ch") -> torch.nn.Module:
    if model_path is not None and not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    if model_lang == "ch":
        n_class = 6625
    elif model_lang == "en":
        n_class = 97
    else:
        raise ValueError(f"Unsupported OCR model_lang: {model_lang}")

    rec_config = edict(
        in_channels=3,
        backbone=edict(type="MobileNetV1Enhance", scale=0.5, last_conv_stride=[1, 2], last_pool_type="avg"),
        neck=edict(type="SequenceEncoder", encoder_type="svtr", dims=64, depth=2, hidden_dims=120, use_guide=True),
        head=edict(type="CTCHead", fc_decay=0.00001, out_channels=n_class, return_feats=True),
    )

    rec_model = RecModel(rec_config)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location="cpu")
        rec_model.load_state_dict(state_dict)
    return rec_model.eval()


class TextRecognizer:
    def __init__(
        self,
        rec_image_shape: str,
        rec_batch_num: int,
        rec_char_dict_path: str,
        predictor: torch.nn.Module,
        device: torch.device,
        use_fp16: bool = False,
    ):
        self.rec_image_shape = [int(v) for v in rec_image_shape.split(",")]
        self.rec_batch_num = rec_batch_num
        self.predictor = predictor
        self.device = device
        self.chars = self.get_char_dict(rec_char_dict_path)
        self.use_fp16 = use_fp16

    def resize_norm_img(self, img: torch.Tensor, max_wh_ratio: float) -> torch.Tensor:
        img_c, img_h, img_w = self.rec_image_shape
        assert img_c == img.shape[0]
        img_w = int(img_h * max_wh_ratio)

        h, w = img.shape[1:]
        ratio = w / float(h)
        if int(np.ceil(img_h * ratio)) > img_w:
            resized_w = img_w
        else:
            resized_w = int(np.ceil(img_h * ratio))

        resized_image = F.interpolate(
            img.unsqueeze(0),
            size=(img_h, resized_w),
            mode="bilinear",
            align_corners=True,
        )
        resized_image /= 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        padding = torch.zeros((img_c, img_h, img_w), dtype=torch.float32, device=img.device)
        padding[:, :, 0:resized_w] = resized_image[0]
        return padding

    def pred_imglist(self, img_list: List[torch.Tensor]) -> torch.Tensor:
        img_num = len(img_list)
        if img_num == 0:
            raise ValueError("img_list is empty.")

        width_list = [img.shape[2] / float(img.shape[1]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        batch_num = self.rec_batch_num
        preds_all: List[torch.Tensor] = [None] * img_num

        img_c, img_h, img_w = self.rec_image_shape
        max_wh_ratio = img_w / img_h

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []

            for ino in range(beg_img_no, end_img_no):
                idx = indices[ino]
                img = img_list[idx]
                h, w = img.shape[1:]
                if h > w * 1.2:
                    img = torch.transpose(img, 1, 2).flip(dims=[1])
                    img_list[idx] = img

            for ino in range(beg_img_no, end_img_no):
                idx = indices[ino]
                norm_img = self.resize_norm_img(img_list[idx], max_wh_ratio)
                if self.use_fp16:
                    norm_img = norm_img.half()
                norm_img_batch.append(norm_img.unsqueeze(0))

            norm_img_batch_tensor = torch.cat(norm_img_batch, dim=0).to(self.device)
            with torch.no_grad():
                preds = self.predictor(norm_img_batch_tensor)

            for rno in range(preds["ctc"].shape[0]):
                idx = indices[beg_img_no + rno]
                preds_all[idx] = preds["ctc"][rno].detach().float().cpu()

        return torch.stack(preds_all, dim=0)

    def get_char_dict(self, character_dict_path: str) -> List[str]:
        character_str = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                character_str.append(line)
        dict_character = list(character_str)
        dict_character = ["sos"] + dict_character + [" "]
        return dict_character

    def get_text(self, order: np.ndarray) -> str:
        char_list = [self.chars[text_id] for text_id in order]
        return "".join(char_list)

    def decode(self, mat: torch.Tensor):
        text_index = mat.detach().cpu().numpy().argmax(axis=1)
        ignored_tokens = [0]
        selection = np.ones(len(text_index), dtype=bool)
        selection[1:] = text_index[1:] != text_index[:-1]
        for ignored_token in ignored_tokens:
            selection &= text_index != ignored_token
        return text_index[selection], np.where(selection)[0]


def ocr_text(text_recognizer: TextRecognizer, image_bgr) -> str:
    if image_bgr is None or image_bgr.size == 0:
        return ""

    img_tensor = torch.from_numpy(image_bgr).permute(2, 0, 1).float()
    preds = text_recognizer.pred_imglist([img_tensor])
    preds_prob = preds.softmax(dim=2)

    order, _ = text_recognizer.decode(preds_prob[0])
    pred_text = text_recognizer.get_text(order)
    return pred_text.strip()


def recognize_masked_text(
    image_bgr,
    mask_gray,
    text_recognizer: TextRecognizer,
    num_regions: int,
    min_component_area: int,
) -> str:
    region_masks = extract_region_masks(mask_gray, num_regions=num_regions, min_component_area=min_component_area)
    region_texts = []
    for region_mask in region_masks:
        crop = crop_with_component_mask(image_bgr, region_mask)
        text = ocr_text(text_recognizer, crop)
        if text:
            region_texts.append(text)
    return " ".join(region_texts).strip()


def compute_metrics(pred_rec_text: str, gt_text: str):
    pred_norm = normalize_text(pred_rec_text)
    gt_norm = normalize_text(gt_text)
    sen_acc = 1 if pred_norm == gt_norm else 0
    ned = 1 - distance(pred_norm, gt_norm) / max(len(pred_norm), len(gt_norm), 1)

    pred_set = set(pred_norm)
    gt_set = set(gt_norm)
    tp = len(pred_set & gt_set)
    if not pred_set and not gt_set:
        precision = 1.0
        recall = 1.0
        f_score = 1.0
    else:
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return sen_acc, ned, precision, recall, f_score


@dataclass
class EvalJob:
    infer_root: Path
    csv_path: Path
    output_json: Path
    start_index: int
    end_index: int


def collect_jobs(args: argparse.Namespace) -> List[EvalJob]:
    jobs: List[EvalJob] = []

    if args.job_with_range:
        for infer_root_str, csv_path_str, output_json_str, start_str, end_str in args.job_with_range:
            jobs.append(
                EvalJob(
                    infer_root=Path(infer_root_str),
                    csv_path=Path(csv_path_str),
                    output_json=Path(output_json_str),
                    start_index=int(start_str),
                    end_index=int(end_str),
                )
            )

    if args.job:
        for infer_root_str, csv_path_str, output_json_str in args.job:
            jobs.append(
                EvalJob(
                    infer_root=Path(infer_root_str),
                    csv_path=Path(csv_path_str),
                    output_json=Path(output_json_str),
                    start_index=args.start_index,
                    end_index=args.end_index,
                )
            )

    if not jobs:
        if args.csv_path is None or args.infer_root is None:
            raise ValueError(
                "Provide at least one of: --job, --job_with_range, or both --infer_root and --csv_path."
            )
        infer_root = Path(args.infer_root)
        csv_path = Path(args.csv_path)
        output_json = Path(args.output_json) if args.output_json else infer_root / "ocr_eval_results.json"
        jobs.append(
            EvalJob(
                infer_root=infer_root,
                csv_path=csv_path,
                output_json=output_json,
                start_index=args.start_index,
                end_index=args.end_index,
            )
        )

    return jobs


def eval_one_job(
    job: EvalJob,
    args: argparse.Namespace,
    text_recognizer: TextRecognizer,
) -> Dict[str, float]:
    csv_path = job.csv_path
    infer_root = job.infer_root
    output_json = job.output_json

    if not csv_path.exists():
        raise FileNotFoundError(f"csv_path not found: {csv_path}")
    if not infer_root.exists():
        raise FileNotFoundError(f"infer_root not found: {infer_root}")

    output_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    end_index = len(df) if job.end_index < 0 else min(job.end_index, len(df))
    start_index = max(0, job.start_index)
    df = df.iloc[start_index:end_index].reset_index(drop=True)

    results: Dict[str, Dict] = {}
    sen_acc_sum = 0.0
    ned_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    f_score_sum = 0.0
    valid_count = 0
    missing_count = 0

    progress_desc = f"OCR eval ({infer_root.name})"
    for _, row in tqdm(df.iterrows(), total=len(df), desc=progress_desc):
        _, sample_dir = resolve_sample_dir(
            infer_root=infer_root,
            row_video=row["video"],
            csv_path=str(csv_path),
            args=args,
        )
        infer_img_path = resolve_named_image(sample_dir, args.pred_image_name)
        gt_img_path = resolve_named_image(sample_dir, args.gt_image_name)
        mask_img_path = resolve_named_image(sample_dir, args.mask_image_name)

        gt_text = safe_text(row.get("gt_text", ""))
        entry = {
            "pred_rec_text": "",
            "gt_rec_text": "",
            "gt_text": gt_text,
            "sen_acc": 0,
            "ned": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f_score": 0.0,
        }

        if not infer_img_path.exists() or not gt_img_path.exists() or not mask_img_path.exists():
            missing_count += 1
            results[str(infer_img_path)] = entry
            continue

        pred_img = cv2.imread(str(infer_img_path), cv2.IMREAD_COLOR)
        gt_img = cv2.imread(str(gt_img_path), cv2.IMREAD_COLOR)
        mask_img = cv2.imread(str(mask_img_path), cv2.IMREAD_GRAYSCALE)

        if pred_img is None or gt_img is None or mask_img is None:
            missing_count += 1
            results[str(infer_img_path)] = entry
            continue

        gt_rec_text = recognize_masked_text(
            gt_img,
            mask_img,
            text_recognizer,
            num_regions=args.num_regions,
            min_component_area=args.min_component_area,
        )

        if 'real' not in str(csv_path) and normalize_text(gt_rec_text) != normalize_text(gt_text): # 必须ocr模型是准确的才记录
            missing_count += 1
            results[str(infer_img_path)] = entry
            continue

        pred_rec_text = recognize_masked_text(
            pred_img,
            mask_img,
            text_recognizer,
            num_regions=args.num_regions,
            min_component_area=args.min_component_area,
        )

        sen_acc, ned, precision, recall, f_score = compute_metrics(pred_rec_text, gt_text)
        entry.update(
            {
                "pred_rec_text": pred_rec_text,
                "gt_rec_text": gt_rec_text,
                "sen_acc": sen_acc,
                "ned": ned,
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }
        )

        results[str(infer_img_path)] = entry
        sen_acc_sum += sen_acc
        ned_sum += ned
        precision_sum += precision
        recall_sum += recall
        f_score_sum += f_score
        valid_count += 1

    mean_sen_acc = sen_acc_sum / valid_count if valid_count else 0.0
    mean_ned = ned_sum / valid_count if valid_count else 1.0
    mean_precision = precision_sum / valid_count if valid_count else 0.0
    mean_recall = recall_sum / valid_count if valid_count else 0.0
    mean_f_score = f_score_sum / valid_count if valid_count else 0.0
    final_metrics = {
        "total_rows": len(df),
        "valid_rows": valid_count,
        "missing_rows": missing_count,
        "mean_sen_acc": mean_sen_acc,
        "mean_ned": mean_ned,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f_score": mean_f_score,
    }

    output_results = dict(results)
    output_results["__final_metrics__"] = final_metrics

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)

    return final_metrics


def main() -> None:
    args = parse_args()
    jobs = collect_jobs(args)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if args.use_gpu and device.type != "cuda":
        print("[WARN] --use_gpu is set, but CUDA is unavailable. Falling back to CPU.")

    predictor = create_predictor(args.rec_model_path, model_lang=args.model_lang).to(device).eval()
    use_fp16 = args.use_fp16 and device.type == "cuda"
    if args.use_fp16 and not use_fp16:
        print("[WARN] --use_fp16 is ignored because CUDA is unavailable.")
    if use_fp16:
        predictor = predictor.half()

    text_recognizer = TextRecognizer(
        rec_image_shape=args.rec_image_shape,
        rec_batch_num=args.rec_batch_num,
        rec_char_dict_path=args.rec_char_dict_path,
        predictor=predictor,
        device=device,
        use_fp16=use_fp16,
    )

    all_job_metrics: List[Dict[str, float]] = []
    for idx, job in enumerate(jobs):
        print("")
        print(f"[JOB {idx + 1}/{len(jobs)}] infer_root={job.infer_root}")
        print(f"[JOB {idx + 1}/{len(jobs)}] csv_path={job.csv_path}")
        print(f"[JOB {idx + 1}/{len(jobs)}] output_json={job.output_json}")
        print(f"[JOB {idx + 1}/{len(jobs)}] rows=[{job.start_index},{job.end_index})")

        metrics = eval_one_job(
            job=job,
            args=args,
            text_recognizer=text_recognizer,
        )
        all_job_metrics.append(metrics)

        print(f"Saved result json: {job.output_json}")
        print(f"Total rows: {metrics['total_rows']}")
        print(f"Valid rows: {metrics['valid_rows']}")
        print(f"Missing rows: {metrics['missing_rows']}")
        print(f"Mean sen_acc: {metrics['mean_sen_acc']:.4f}")
        print(f"Mean ned: {metrics['mean_ned']:.4f}")
        print(f"Mean precision: {metrics['mean_precision']:.4f}")
        print(f"Mean recall: {metrics['mean_recall']:.4f}")
        print(f"Mean f_score: {metrics['mean_f_score']:.4f}")

    if len(all_job_metrics) > 1:
        total_rows = sum(m["total_rows"] for m in all_job_metrics)
        valid_rows = sum(m["valid_rows"] for m in all_job_metrics)
        missing_rows = sum(m["missing_rows"] for m in all_job_metrics)
        mean_sen_acc = (
            sum(m["mean_sen_acc"] * m["valid_rows"] for m in all_job_metrics) / max(valid_rows, 1)
        )
        mean_ned = sum(m["mean_ned"] * m["valid_rows"] for m in all_job_metrics) / max(valid_rows, 1)
        mean_precision = (
            sum(m["mean_precision"] * m["valid_rows"] for m in all_job_metrics) / max(valid_rows, 1)
        )
        mean_recall = sum(m["mean_recall"] * m["valid_rows"] for m in all_job_metrics) / max(valid_rows, 1)
        mean_f_score = sum(m["mean_f_score"] * m["valid_rows"] for m in all_job_metrics) / max(valid_rows, 1)
        print("")
        print("========== OCR Eval Summary ==========")
        print(f"jobs: {len(all_job_metrics)}")
        print(f"total_rows: {total_rows}")
        print(f"valid_rows: {valid_rows}")
        print(f"missing_rows: {missing_rows}")
        print(f"weighted_mean_sen_acc: {mean_sen_acc:.4f}")
        print(f"weighted_mean_ned: {mean_ned:.4f}")
        print(f"weighted_mean_precision: {mean_precision:.4f}")
        print(f"weighted_mean_recall: {mean_recall:.4f}")
        print(f"weighted_mean_f_score: {mean_f_score:.4f}")
        print("======================================")


if __name__ == "__main__":
    main()
