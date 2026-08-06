import argparse
import json
import shutil
from pathlib import Path

import torch

from eval_baseline import (
    FVDContext,
    build_ocr_recognizer,
    ensure_dir,
    evaluate_one_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_dataset_specs(value):
    datasets = []
    for raw in value.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        csv_path = Path(raw)
        datasets.append(csv_path)
    if not datasets:
        raise ValueError("No datasets specified")
    return datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SteerVTE outputs for multiple CSV datasets.")
    parser.add_argument("--dataset_specs", required=True, help="csv;csv;csv")
    parser.add_argument("--infer_root", required=True, help="Root containing one output directory per CSV.")
    parser.add_argument("--csv_root", default=".", help="Root for relative media paths in CSV files.")
    parser.add_argument("--result_root", required=True, help="Directory for one JSON result per CSV.")
    parser.add_argument("--tmp_root", default="outputs/eval_tmp")
    parser.add_argument("--src_name_suffix", default="tgt")
    parser.add_argument("--prepare_workers", type=int, default=8)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--num_regions", type=int, default=1)
    parser.add_argument("--min_component_area", type=int, default=32)
    parser.add_argument("--frame_number", type=int, default=20)
    parser.add_argument("--square_size", type=int, default=224)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument(
        "--rec_model_path",
        default=str(PROJECT_ROOT / "models/SteerVTE/ocr_weights/ppv3_rec.pth"),
    )
    parser.add_argument(
        "--rec_char_dict_path",
        default=str(PROJECT_ROOT / "models/SteerVTE/ocr_weights/ppocr_keys_v1.txt"),
    )
    parser.add_argument("--rec_image_shape", default="3,48,320")
    parser.add_argument("--rec_batch_num", type=int, default=64)
    parser.add_argument("--model_lang", choices=["ch", "en"], default="ch")
    parser.add_argument("--fid_batch_size", type=int, default=1)
    parser.add_argument("--fid_dims", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = parse_dataset_specs(args.dataset_specs)
    infer_root = Path(args.infer_root)
    csv_root = Path(args.csv_root)
    result_root = Path(args.result_root)
    tmp_root = Path(args.tmp_root)

    for csv_path in datasets:
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        edited_dir = infer_root / csv_path.stem
        if not edited_dir.is_dir():
            raise FileNotFoundError(f"Inference output directory not found: {edited_dir}")

    ensure_dir(result_root)
    ensure_dir(tmp_root)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    recognizer = build_ocr_recognizer(args, device)
    fvd_ctx = FVDContext(model=None)

    print("========== Multi-CSV Eval ==========")
    print(f"infer_root: {infer_root}")
    print(f"csv_root:   {csv_root}")
    print(f"result_root:{result_root}")
    print(f"device:     {device}")
    print(f"datasets:   {len(datasets)}")
    print("=====================================")

    for csv_path in datasets:
        dataset_name = csv_path.stem
        edited_dir = infer_root / dataset_name
        dataset_tmp_root = tmp_root / dataset_name
        if dataset_tmp_root.exists():
            shutil.rmtree(dataset_tmp_root)
        ensure_dir(dataset_tmp_root)

        result = evaluate_one_split(
            split_name="all",
            src_dir=Path("."),
            edited_dir=edited_dir,
            csv_path=csv_path,
            base_index=0,
            limit=10**9,
            args=args,
            device=device,
            recognizer=recognizer,
            method_tmp_dir=dataset_tmp_root,
            fvd_ctx=fvd_ctx,
            src_from_csv=True,
            csv_root=csv_root,
            pred_name_width=0,
            pred_index_base=1,
            skip_missing_pred=args.skip_missing,
        )
        result["meta"]["dataset_name"] = dataset_name
        result["meta"]["csv_path"] = str(csv_path)
        result["meta"]["infer_dir"] = str(edited_dir)
        result["meta"]["csv_root"] = str(csv_root)

        output_json = result_root / f"{dataset_name}.json"
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved eval json: {output_json}")


if __name__ == "__main__":
    main()
