import os
import pandas as pd
import torch
from PIL import Image

from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video

LORA_PATH = "models/SteerVTE/SteerVTE.safetensors"
CSV_PATH = "demo_data/infer.csv"
OUTPUT_DIR = "outputs/demo"

def _extract_prefixed(state_dict, prefixes):
    result = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                result[key[len(prefix):]] = value
                break
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        style_encoder_type="qwen_vlm",
        proj_type="linear",
    )

    ckpt = load_state_dict(LORA_PATH, device="cpu")
    pipe.load_lora(pipe.vace, LORA_PATH, alpha=1)

    style_proj = getattr(pipe.style_image_encoder, "style_proj_out", None) or getattr(pipe.style_image_encoder, "proj", None)
    style_state = _extract_prefixed(ckpt, ("pipe.style_image_encoder.style_proj_out.", "style_image_encoder.style_proj_out.", "pipe.style_image_encoder.proj.", "style_image_encoder.proj."))
    if style_proj is not None and style_state:
        style_proj.load_state_dict(style_state, strict=False)

    ocr_state = _extract_prefixed(ckpt, ("pipe.ocr_image_encoder.proj.", "ocr_image_encoder.proj."))
    if ocr_state:
        pipe.ocr_image_encoder.proj.load_state_dict(ocr_state, strict=False)

    pipe.to("cuda")
    for idx, row in pd.read_csv(CSV_PATH).iterrows():
        video = VideoData(row["vace_video"])
        mask_video = VideoData(row["vace_video_mask"])
        h, w = video.shape()
        control = [video[i] for i in range(49)]
        mask = [mask_video[i] for i in range(49)]

        frames = pipe(
            prompt=row["prompt"],
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            vace_video=control,
            vace_video_mask=mask,
            vace_reference_image=Image.open(row["vace_reference_image"]).convert("RGB"),
            vace_style_image=Image.open(row["vace_style_image"]).convert("RGB"),
            gt_text=row["gt_text"],
            src_text=row["src_text"],
            num_frames=49,
            num_inference_steps=20,
            vace_preprocess_type="remain_all",
            height=h,
            width=w,
            seed=1,
            tiled=True,
        )

        out_path = os.path.join(OUTPUT_DIR, f"{idx + 1}.mp4")
        save_video(frames, out_path, fps=16, quality=5)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()