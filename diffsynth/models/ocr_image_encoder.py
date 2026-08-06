import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from easydict import EasyDict as edict
except Exception:
    from types import SimpleNamespace as edict

from PIL import Image, ImageDraw, ImageFont

from diffsynth.models.recognizer import TextRecognizer, create_predictor

DEFAULT_OCR_CHARSET = " '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DEFAULT_OCR_MODEL_PATH = "models/SteerVTE/ocr_weights/ppv3_rec.pth"
DEFAULT_OCR_CHAR_DICT_PATH = "models/SteerVTE/ocr_weights/ppocr_keys_v1.txt"
DEFAULT_FONT_PATH = "models/SteerVTE/ocr_weights/arialuni.ttf"


class OCRImageEncoder(nn.Module):
    def __init__(
        self,
        rec_model_path: Optional[str] = None,
        rec_char_dict_path: Optional[str] = None,
        model_lang: str = "ch",
        font_path: Optional[str] = None,
        canvas_size: int = 80,
        font_size: int = 60,
        rec_batch_num: int = 64,
        in_dim: int = 1280,
        proj_dim: int = 5120,
        use_fp16: bool = False,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        emb_cache_path: str = "models/SteerVTE/ocr_weights/ocr_char_emb.pt",
        charset: str = DEFAULT_OCR_CHARSET,
        rebuild_cache_on_init: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.canvas_size = int(canvas_size)
        self.font_size = int(font_size)
        self.proj_dim = int(proj_dim)
        self.in_dim = int(in_dim)
        self.emb_cache_path = emb_cache_path
        self.charset = charset

        self.ocr_chars: List[str] = []
        self.ocr_char2idx: Dict[str, int] = {}
        self.ocr_char_emb: Optional[torch.Tensor] = None

        rec_model_path = rec_model_path or str(DEFAULT_OCR_MODEL_PATH)
        rec_char_dict_path = rec_char_dict_path or str(DEFAULT_OCR_CHAR_DICT_PATH)
        self.font_path = font_path or str(DEFAULT_FONT_PATH)
        self.font = ImageFont.truetype(self.font_path, self.font_size)

        predictor = create_predictor(str(rec_model_path), model_lang=model_lang)
        if isinstance(predictor, torch.nn.Module):
            predictor = predictor.to(self.device, dtype=self.dtype)
            predictor.eval()
            for param in predictor.parameters():
                param.requires_grad = False

        args = edict()
        args.rec_image_shape = "3, 80, 80"
        args.rec_batch_num = int(rec_batch_num)
        args.rec_char_dict_path = str(rec_char_dict_path)
        args.use_fp16 = bool(use_fp16)
        self.recognizer = TextRecognizer(args, predictor)

        # self.proj = nn.Sequential(nn.Linear(self.in_dim, self.proj_dim), nn.LayerNorm(self.proj_dim))
        # self.proj = nn.Sequential(nn.LazyLinear(self.proj_dim), nn.LayerNorm(self.proj_dim))
        self.proj = nn.Sequential(
                nn.LayerNorm(self.in_dim),
                nn.Linear(self.in_dim, self.proj_dim, bias=True),
            )
        self.proj.apply(self._init_weights)
        self.to(self.device, dtype=self.dtype)

        # Load OCR embedding table at init and move to target device.
        self.get_ocr_emb(
            charset=self.charset,
            cache_path=self.emb_cache_path,
            force_rebuild=rebuild_cache_on_init,
        )
    
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

    def _render_char_tensor(self, char: str) -> torch.Tensor:
        image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        draw = ImageDraw.Draw(image)

        try:
            left, top, right, bottom = draw.textbbox((0, 0), char, font=self.font, anchor="lt")
            text_w = right - left
            text_h = bottom - top
        except Exception:
            text_w, text_h = self.font.getsize(char)
            left, top = 0, 0

        x = (self.canvas_size - text_w) / 2
        y = (self.canvas_size - text_h) / 2
        draw.text((x, y), char, fill=255, font=self.font, anchor="lt")

        # image.save(f"tmp/ref_img/{char}.png")

        glyph = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0)
        return glyph.to(self.device)

    def render_text(self, text: str) -> Tuple[List[str], List[torch.Tensor]]:
        chars = list(text)
        glyph_tensors = [self._render_char_tensor(char) for char in chars]
        return chars, glyph_tensors

    def get_recog_emb(self, img_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(img_list) == 0:
            return torch.empty((0, 0), dtype=self.dtype, device=self.device)

        repeat = 3 if img_list[0].shape[0] == 1 else 1
        rgb_img_list = [img.repeat(repeat, 1, 1) if repeat == 3 else img for img in img_list]

        self.recognizer.predictor.eval()
        with torch.no_grad():
            _, preds_neck = self.recognizer.pred_imglist(rgb_img_list, show_debug=False, norm=True)

            # preds, preds_neck = self.recognizer.pred_imglist(rgb_img_list, show_debug=True, norm=True)
            # preds_all = preds.softmax(dim=2)
            # for i in range(len(preds_all)):
            #     pred_prob = preds_all[i]
            #     order, _ = self.recognizer.decode(pred_prob)
            #     pred_text = self.recognizer.get_text(order)
            #     print(f"{i}, Pred text: {pred_text}")
        return preds_neck.reshape(preds_neck.shape[0], -1).to(device=self.device, dtype=self.dtype)

    def get_ocr_emb(
        self,
        charset: str = DEFAULT_OCR_CHARSET,
        cache_path: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        cache_file = Path(cache_path or self.emb_cache_path)
        if cache_file.exists() and not force_rebuild:
            cache_data = torch.load(cache_file, map_location="cpu")
            self.ocr_chars = list(cache_data["chars"])
            self.ocr_char2idx = dict(cache_data["char2idx"])
            self.ocr_char_emb = cache_data["ocr_emb"].to(device=self.device, dtype=self.dtype)
            return self.ocr_char_emb, self.ocr_char2idx

        chars = list(charset)
        _, glyph_tensors = self.render_text(charset)
        neck_features = self.get_recog_emb(glyph_tensors).detach().cpu()

        char2idx = {char: idx for idx, char in enumerate(chars)}
        cache_data = {"chars": chars, "char2idx": char2idx, "ocr_emb": neck_features}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache_data, cache_file)

        self.ocr_chars = chars
        self.ocr_char2idx = char2idx
        self.ocr_char_emb = neck_features.to(device=self.device, dtype=self.dtype)
        return self.ocr_char_emb, self.ocr_char2idx

    def _append_unseen_chars(self, unseen_chars: List[str]) -> None:
        if len(unseen_chars) == 0:
            return

        _, glyph_tensors = self.render_text("".join(unseen_chars))
        new_neck = self.get_recog_emb(glyph_tensors)

        if self.ocr_char_emb is None:
            self.ocr_char_emb = new_neck
        else:
            self.ocr_char_emb = torch.cat([self.ocr_char_emb, new_neck], dim=0)

        start_idx = len(self.ocr_chars)
        for i, char in enumerate(unseen_chars):
            self.ocr_chars.append(char)
            self.ocr_char2idx[char] = start_idx + i

    def forward(self, text: str, return_chars: bool = False):
        chars = list(text)
        if len(chars) == 0:
            empty_features = torch.empty((0, self.proj_dim), dtype=self.dtype, device=self.device)
            return (empty_features, chars) if return_chars else empty_features

        # In-vocab chars use cached embedding; OOV chars fallback to OCR inference.
        unseen_chars = [char for char in chars if char not in self.ocr_char2idx]
        if len(unseen_chars) > 0:
            unique_unseen = list(dict.fromkeys(unseen_chars))
            self._append_unseen_chars(unique_unseen)

        indices = [self.ocr_char2idx[char] for char in chars]
        index_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        neck_features = self.ocr_char_emb[index_tensor]
        proj_dtype = next(self.proj.parameters()).dtype
        neck_features = neck_features.to(dtype=proj_dtype)
        features = self.proj(neck_features)

        return (features, chars) if return_chars else features


def _parse_args():
    parser = argparse.ArgumentParser(description="Character-level glyph embedding extractor")
    parser.add_argument("--text", type=str, default="Free")
    parser.add_argument("--charset", type=str, default=DEFAULT_OCR_CHARSET)
    parser.add_argument("--cache-path", type=str, default="models/SteerVTE/ocr_weights/ocr_char_emb.pt")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--model-lang", type=str, default="ch", choices=["ch", "en"])
    parser.add_argument("--rec-model-path", type=str, default=None)
    parser.add_argument("--rec-char-dict-path", type=str, default=None)
    parser.add_argument("--font-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--proj-dim", type=int, default=5120)
    return parser.parse_args()


def main():
    args = _parse_args()
    encoder = OCRImageEncoder(
        rec_model_path=args.rec_model_path,
        rec_char_dict_path=args.rec_char_dict_path,
        model_lang=args.model_lang,
        font_path=args.font_path,
        proj_dim=args.proj_dim,
        device=args.device,
        dtype=torch.bfloat16,
        emb_cache_path=args.cache_path,
        charset=args.charset,
        rebuild_cache_on_init=args.rebuild_cache,
    )

    print(f"Cache loaded: {args.cache_path}")
    print(f"Cached charset size: {len(encoder.ocr_chars)}")

    features, chars = encoder(args.text, return_chars=True)
    print(f"Input text: {args.text}")
    print(f"Chars: {chars}")
    print(f"Embedding shape: {tuple(features.shape)}")
    print(f"Embedding dtype/device: {features.dtype}/{features.device}")


if __name__ == "__main__":
    main()
