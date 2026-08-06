import os
import torch
import torch.nn as nn
from typing import List, Optional, Sequence, Union
from PIL import Image

from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffsynth.models.qwen3_vl_embedding import Qwen3VLEmbedder

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None


def _pad_feature_list(features: List[torch.Tensor], device, dtype) -> torch.Tensor:
    if not features:
        return torch.zeros((0, 0, 0), device=device, dtype=dtype)

    max_len = max(f.shape[0] for f in features)
    feat_dim = features[0].shape[1]
    batch = len(features)
    padded = torch.zeros((batch, max_len, feat_dim), device=device, dtype=dtype)
    for i, f in enumerate(features):
        if f.numel() == 0:
            continue
        padded[i, : f.shape[0], :] = f
    return padded


def _normalize_proj_type(proj_type: Optional[str]) -> str:
    proj_type = (proj_type or "linear").lower()
    if proj_type not in ("linear", "mlp"):
        raise ValueError(f"Unsupported proj_type={proj_type}. Available: ['linear', 'mlp']")
    return proj_type


def _build_style_proj(in_dim: int, out_dim: int, proj_type: str) -> nn.Module:
    if proj_type == "linear":
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim, bias=True),
        )
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, in_dim, bias=True),
        nn.GELU(),
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim, bias=True),
    )


class QwenVitStyleEncoder(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):

    @register_to_config
    def __init__(
        self,
        model_id: str = "models/Qwen/Qwen2.5-VL-3B-Instruct",
        safetensors_path: str = "models/qwenvl/qwen2_5_vl_vit.safetensors",
        cache_dir: Optional[str] = None,
        out_dim: Optional[int] = None,
        proj_type: str = "linear",
    ):
        super().__init__()

        self.pretrained_module_names = ["vit"]

        config = Qwen2_5_VLConfig.from_pretrained(model_id, cache_dir=cache_dir)
        vision_config = config.vision_config
        self.vit = Qwen2_5_VisionTransformerPretrainedModel(vision_config)

        if safetensors_path is not None:
            state_dict = load_file(safetensors_path)
            self.vit.load_state_dict(state_dict, strict=False)

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=False)

        vit_hidden_dim = vision_config.out_hidden_size
        self.out_dim = out_dim
        self.proj_type = _normalize_proj_type(proj_type)
        if self.out_dim is not None:
            self.style_proj_out = _build_style_proj(vit_hidden_dim, self.out_dim, self.proj_type)
            self.style_proj_out.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         # nn.init.xavier_uniform_(module.weight)
    #         nn.init.xavier_uniform_(module.weight, gain=0.5)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.LayerNorm):
    #         if module.weight is not None:
    #             nn.init.ones_(module.weight)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for key in list(sd.keys()):
            if any([key.startswith(module_name) for module_name in self.pretrained_module_names]):
                sd.pop(key)
        return sd

    def forward(
        self,
        style_images: List[Union[Image.Image, List[Image.Image]]],
        *args, **kwargs,
    ) -> torch.Tensor:
        """Return:
        style_embeds: (b, S, D') or (b, S, D) if out_dim is None
        """
        def _normalize_batch(images: Sequence[Union[Image.Image, List[Image.Image]]]) -> List[List[Image.Image]]:
            normalized = []
            for item in images:
                if isinstance(item, list):
                    normalized.append(item)
                else:
                    normalized.append([item])
            return normalized

        style_batch = _normalize_batch(style_images)

        style_counts = [len(patches) for patches in style_batch]
        flat_style = [p for patches in style_batch for p in patches]

        if len(flat_style) == 0:
            return torch.zeros((len(style_batch), 0, 0), device=self.vit.device, dtype=self.vit.dtype)

        inputs = self.processor(images=flat_style, text=[""] * len(flat_style), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.vit.device, dtype=self.vit.dtype)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device=self.vit.device)

        outputs = self.vit(pixel_values, grid_thw=image_grid_thw)
        if isinstance(outputs, torch.Tensor):
            style_embeds_flat = outputs
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            style_embeds_flat = outputs.last_hidden_state
        elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            style_embeds_flat = outputs.image_embeds
        else:
            raise ValueError("Qwen ViT did not return image embeddings.")

        if style_embeds_flat.dim() == 2:
            style_embeds_flat = style_embeds_flat.unsqueeze(0)

        style_embeds_list = []
        offset = 0
        for count in style_counts:
            if count == 0:
                style_embeds_list.append(style_embeds_flat.new_zeros((0, style_embeds_flat.shape[-1])))
            else:
                style_embeds_list.append(style_embeds_flat[offset : offset + count].reshape(-1, style_embeds_flat.shape[-1]))
            offset += count

        style_embeds = _pad_feature_list(
            style_embeds_list,
            device=style_embeds_flat.device,
            dtype=style_embeds_flat.dtype,
        )

        if self.out_dim is not None:
            style_embeds = self.style_proj_out(style_embeds)

        return style_embeds


class QwenVlmStyleEncoder(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):

    @register_to_config
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        cache_dir: Optional[str] = None,
        out_dim: Optional[int] = None,
        system_prompt: str = "You are a helpful assistant that excels at extracting visual features of text from images.",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        attn_implementation: Optional[str] = "flash_attention_2",
        device_map: Optional[Union[str, dict]] = None,
        use_fast: bool = False,
        freeze_vlm: bool = True,
        use_cache: bool = False,
        proj_type: str = "linear",
    ):
        super().__init__()

        if process_vision_info is None:
            raise ImportError("qwen_vl_utils.process_vision_info is required for QwenVlmStyleEncoder.")

        self.pretrained_module_names = ["model"]

        if device_map is None and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_map = {"": f"cuda:{local_rank}"}
            print(f"QwenVlmStyleEncoder device_map: {device_map}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )

        # print("🚀 QwenVlmStyleEncoder model:", self.model.config)

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = use_cache
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, use_fast=use_fast)

        self.system_prompt = system_prompt
        self.freeze_vlm = freeze_vlm
        self.use_cache = use_cache

        if freeze_vlm:
            self.model.requires_grad_(False)
            self.model.eval()

        hidden_dim = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_dim is None:
            raise ValueError("QwenVlmStyleEncoder requires model.config.text_config.hidden_size to build projection.")
        self.style_dim = hidden_dim * 2

        self.out_dim = out_dim
        self.proj_type = _normalize_proj_type(proj_type)
        if self.out_dim is not None:
            self.style_proj_out = _build_style_proj(self.style_dim, self.out_dim, self.proj_type)
            self.style_proj_out.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_uniform_(module.weight, gain=0.5)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.LayerNorm):
    #         if module.weight is not None:
    #             nn.init.ones_(module.weight)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for key in list(sd.keys()):
            if any([key.startswith(module_name) for module_name in self.pretrained_module_names]):
                sd.pop(key)
        return sd

    def _get_device(self):
        return next(self.model.parameters()).device

    def _cast_floats(self, batch, dtype):
        for key, value in batch.items():
            if torch.is_floating_point(value):
                batch[key] = value.to(dtype=dtype)
        return batch

    def forward(
        self,
        style_images: List[Union[Image.Image, List[Image.Image]]],
        gt_text: Union[str, Sequence[str]],
        *args, **kwargs,
    ) -> torch.Tensor:
        """Return:
        style_embeds: (b, S, D') or (b, S, D) if out_dim is None
        """
        def _normalize_batch(images: Sequence[Union[Image.Image, List[Image.Image]]]) -> List[List[Image.Image]]:
            normalized = []
            for item in images:
                if isinstance(item, list):
                    normalized.append(item)
                else:
                    normalized.append([item])
            return normalized

        style_batch = _normalize_batch(style_images)
        if isinstance(gt_text, str):
            gt_text_batch = [gt_text] * len(style_batch)
        else:
            gt_text_batch = list(gt_text)
            if len(gt_text_batch) != len(style_batch):
                raise ValueError(
                    f"gt_text batch size mismatch: got {len(gt_text_batch)}, expected {len(style_batch)}."
                )

        device = self._get_device()
        dtype = next(self.model.parameters()).dtype

        if len(style_batch) == 0:
            return torch.zeros((0, 0, 0), device=device, dtype=dtype)

        style_embeds_list = []
        for patches, sample_gt_text in zip(style_batch, gt_text_batch):
            if len(patches) == 0:
                style_embeds_list.append(torch.zeros((0, self.style_dim), device=device, dtype=dtype))
                continue

            content = [{"type": "image", "image": img} for img in patches]
            user_prompt = (
                "Describe detailedly the typography color, style, text material, and rendering effects of "
                f"the text regions '{sample_gt_text}' in this image."
            )
            content.append({"type": "text", "text": user_prompt})

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": content})

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            inputs = self._cast_floats(inputs, dtype=dtype)

            if self.freeze_vlm:
                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=self.use_cache,
                    )
            else:
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=self.use_cache,
                )
            h_minus_1 = outputs.hidden_states[-1]
            h_minus_2 = outputs.hidden_states[-2]
            combined_features = torch.cat([h_minus_2, h_minus_1], dim=-1)

            if combined_features.dim() == 3:
                combined_features = combined_features[0]
            if self.freeze_vlm:
                combined_features = combined_features.detach()
            style_embeds_list.append(combined_features)

        style_embeds = _pad_feature_list(
            style_embeds_list,
            device=device,
            dtype=dtype,
        )

        if self.out_dim is not None:
            style_embeds = self.style_proj_out(style_embeds)

        return style_embeds


class QwenVlEmbeddingStyleEncoder(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):

    @register_to_config
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-Embedding-2B",
        cache_dir: Optional[str] = None,
        out_dim: Optional[int] = None,
        user_prompt: str = (
            "Please describe the text area attributes: "
            "text color, background color, stroke, and shadow."
        ),
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        attn_implementation: Optional[str] = "flash_attention_2",
        normalize: bool = True,
        freeze_vlm: bool = True,
        proj_type: str = "linear",
    ):
        super().__init__()

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            cache_dir=cache_dir,
        )
        self.user_prompt = user_prompt
        self.normalize = normalize
        self.freeze_vlm = freeze_vlm

        if freeze_vlm:
            self.embedder.model.requires_grad_(False)
            self.embedder.model.eval()

        self.embed_dim = 2048
        if self.embed_dim is None:
            raise ValueError("QwenVlEmbeddingStyleEncoder requires model.config.hidden_size to build projection.")

        self.out_dim = out_dim
        self.proj_type = _normalize_proj_type(proj_type)
        if self.out_dim is not None:
            self.style_proj_out = _build_style_proj(self.embed_dim, self.out_dim, self.proj_type)
            self.style_proj_out.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _get_device_dtype(self):
        param = next(self.embedder.model.parameters())
        return param.device, param.dtype

    def forward(
        self,
        style_images: List[Union[Image.Image, List[Image.Image]]],
        *args, **kwargs,
    ) -> torch.Tensor:
        """Return:
        style_embeds: (b, S, D') or (b, S, D) if out_dim is None
        """
        def _normalize_batch(images: Sequence[Union[Image.Image, List[Image.Image]]]) -> List[List[Image.Image]]:
            normalized = []
            for item in images:
                if isinstance(item, list):
                    normalized.append(item)
                else:
                    normalized.append([item])
            return normalized

        style_batch = _normalize_batch(style_images)
        device, dtype = self._get_device_dtype()

        if len(style_batch) == 0:
            return torch.zeros((0, 0, 0), device=device, dtype=dtype)

        style_embeds_list = []
        for patches in style_batch:
            if len(patches) == 0:
                style_embeds_list.append(torch.zeros((0, self.embed_dim), device=device, dtype=dtype))
                continue

            inputs = [{"text": self.user_prompt, "image": img} for img in patches]

            if self.freeze_vlm:
                with torch.no_grad():
                    embeds = self.embedder.process(inputs, normalize=self.normalize)
            else:
                embeds = self.embedder.process(inputs, normalize=self.normalize)

            if embeds.dim() == 1:
                embeds = embeds.unsqueeze(0)
            if self.freeze_vlm:
                embeds = embeds.detach()

            style_embeds_list.append(embeds)

        style_embeds = _pad_feature_list(
            style_embeds_list,
            device=device,
            dtype=dtype,
        )

        if self.out_dim is not None:
            style_embeds = self.style_proj_out(style_embeds)

        return style_embeds


if __name__ == "__main__":
    from PIL import Image
    import time

    print("QwenVlmStyleEncoder smoke test starting...")
    t0 = time.time()

    reference_style_image_path = "data/video_text_data/stage2_test_data/pexels_480p_81f/5000-3/8240573_resize1080p/0-81/src_style.png"
    style_images = [[Image.open(reference_style_image_path).convert("RGB")]]
    gt_text = "DIFFSYNTH"

    # fm = QwenVitStyleEncoder(out_dim=1536).to("cuda").to(dtype=torch.bfloat16)
    fm = QwenVlmStyleEncoder(out_dim=1536).to("cuda").to(dtype=torch.bfloat16)
    # fm = QwenVlEmbeddingStyleEncoder(out_dim=1536).to("cuda").to(dtype=torch.bfloat16)

    style_embeds = fm(style_images, gt_text=gt_text)

    print("Done.")
    print("style_embeds:", getattr(style_embeds, "shape", type(style_embeds)))
    print(f"max min value style_embeds: {style_embeds.max().item():.4f} {style_embeds.min().item():.4f}")

    print(f"Elapsed: {time.time()-t0:.2f}s")
