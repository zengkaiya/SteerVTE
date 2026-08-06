import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from PIL import Image

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def prefixed_state(state_dict, prefixes):
    result = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                result[key[len(prefix):]] = value
                break
    return result


def load_pipeline(weight_path, style_encoder_type, proj_type):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        add_ocr_embedding=True,
        add_style_embedding=True,
        style_encoder_type=style_encoder_type,
        proj_type=proj_type,
    )
    pipe.load_lora(pipe.vace, weight_path, alpha=1)

    state = load_state_dict(weight_path, device="cpu")
    style_state = prefixed_state(
        state,
        (
            "pipe.style_image_encoder.style_proj_out.",
            "style_image_encoder.style_proj_out.",
            "pipe.style_image_encoder.proj.",
            "style_image_encoder.proj.",
        ),
    )
    style_proj = getattr(pipe.style_image_encoder, "style_proj_out", None)
    if style_proj is None:
        style_proj = getattr(pipe.style_image_encoder, "proj", None)
    if style_proj is not None and style_state:
        style_proj.load_state_dict(style_state, strict=False)

    ocr_state = prefixed_state(state, ("pipe.ocr_image_encoder.proj.", "ocr_image_encoder.proj."))
    if ocr_state:
        pipe.ocr_image_encoder.proj.load_state_dict(ocr_state, strict=False)
    pipe.to("cuda")
    return pipe


def parse_specs(specs):
    """Parse csv_path|start|end entries produced by infer_bench.sh."""
    datasets = []
    for raw in specs.split(";"):
        parts = raw.strip().split("|")
        if len(parts) != 3:
            raise ValueError(f"Invalid dataset spec: {raw!r}; expected csv|start|end")
        datasets.append((parts[0], int(parts[1]), int(parts[2])))
    return datasets


def resolve_csv_path(path, csv_root):
    path = str(path)
    if os.path.isabs(path) or not csv_root:
        return path
    return os.path.join(csv_root, path)


def run_dataset(pipe, args, csv_path, start, end):
    df = pd.read_csv(csv_path, skiprows=range(1, start + 1), nrows=end - start)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join(args.infer_root, name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{args.gpu_label}] {csv_path}: rows [{start}, {end}), loaded {len(df)}")

    for local_index, (_, row) in enumerate(df.iterrows()):
        row_index = start + local_index
        video = VideoData(resolve_csv_path(row["vace_video"], args.csv_root))
        mask_video = VideoData(resolve_csv_path(row["vace_video_mask"], args.csv_root))
        height, width = video.shape()
        control = [video[i] for i in range(args.num_frames)]
        mask = [mask_video[i] for i in range(args.num_frames)]
        frames = pipe(
            prompt=row["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            vace_video=control,
            vace_video_mask=mask,
            vace_reference_image=Image.open(
                resolve_csv_path(row["vace_reference_image"], args.csv_root)
            ).convert("RGB"),
            vace_style_image=Image.open(
                resolve_csv_path(row["vace_style_image"], args.csv_root)
            ).convert("RGB"),
            gt_text=row["gt_text"],
            src_text=row["src_text"],
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            vace_preprocess_type=args.vace_preprocess_type,
            height=height,
            width=width,
            seed=args.seed,
            tiled=True,
        )
        out_path = os.path.join(output_dir, f"{row_index + 1}.mp4")
        save_video(frames, out_path, fps=16, quality=5)
        print(f"[{args.gpu_label}] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU SteerVTE benchmark inference")
    parser.add_argument("--weight_path", default="models/SteerVTE/SteerVTE.safetensors")
    parser.add_argument("--dataset_specs", required=True)
    parser.add_argument("--csv_root", default=".")
    parser.add_argument("--infer_root", default="outputs/bench")
    parser.add_argument("--vace_preprocess_type", default="remain_all")
    parser.add_argument("--style_encoder_type", default="qwen_vlm")
    parser.add_argument("--proj_type", default="linear")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu_label", default="gpu")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark inference")
    if args.num_frames <= 0 or args.num_inference_steps <= 0:
        raise ValueError("num_frames and num_inference_steps must be positive")
    if not os.path.isfile(args.weight_path):
        raise FileNotFoundError(args.weight_path)
    datasets = parse_specs(args.dataset_specs)
    for csv_path, _, _ in datasets:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)

    print(f"Loading pipeline on {args.gpu_label}...")
    pipe = load_pipeline(args.weight_path, args.style_encoder_type, args.proj_type)
    for csv_path, start, end in datasets:
        if start < end:
            run_dataset(pipe, args, csv_path, start, end)


if __name__ == "__main__":
    main()
