import cv2
import numpy as np
from easydict import EasyDict as edict
from pathlib import Path
import os
import os.path as osp
import sys
import torch
import torch.nn.functional as F
from skimage.transform._geometric import _umeyama as get_sym_mat
try:
    from .recognizer import TextRecognizer, create_predictor
except Exception:
    from recognizer import TextRecognizer, create_predictor
from torchvision.transforms import functional as TF

PRINT_DEBUG = False
DEFAULT_OCR_MODEL_DIR = "models/SteerVTE/ocr_weights/ppv3_rec.pth"
DEFAULT_OCR_CHAR_DICT_PATH = "models/SteerVTE/ocr_weights/ppocr_keys_v1.txt"

def save_tensor_image(tensor, path, rgb=True, normalize=True):
    """
    tensor: (C, H, W)
    """
    img = tensor.detach().float().cpu()

    if normalize:
        minv = img.min()
        maxv = img.max()
        if maxv > minv:
            img = (img - minv) / (maxv - minv)

    img = (img * 255).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).numpy()  # (H, W, C)

    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def read_video_cv2(
    video_path,
    device="cpu",
    dtype=torch.float32,
    frames=None,
    frame_stride=1,
    max_frames=None,
):
    cap = cv2.VideoCapture(video_path)
    frames_out = []

    idx = 0
    keep_count = 0
    frames_set = set(frames) if frames is not None else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frames_set is not None:
            take = idx in frames_set
        else:
            take = (idx % frame_stride == 0)

        if take:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
            frames_out.append(frame)
            keep_count += 1

            if max_frames is not None and keep_count >= max_frames:
                break

        idx += 1

    cap.release()

    if len(frames_out) == 0:
        raise ValueError(f"No frames selected from video: {video_path}")

    video = torch.stack(frames_out).float() / 255.0

    video = video.permute(1, 0, 2, 3).contiguous()

    video = video.to(device=device, dtype=dtype)

    return video

def read_mask_video_cv2(
    video_path,
    device="cpu",
    dtype=torch.float32,
    frames=None,
    frame_stride=1,
    max_frames=None,
    binarize=True,
):
    cap = cv2.VideoCapture(video_path)
    frames_out = []

    idx = 0
    keep_count = 0
    frames_set = set(frames) if frames is not None else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        take = False
        if frames_set is not None:
            take = idx in frames_set
        else:
            take = (idx % frame_stride == 0)

        if take:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = torch.from_numpy(gray)[None]

            if binarize:
                mask = (mask > 0).float()

            frames_out.append(mask)
            keep_count += 1

            if max_frames is not None and keep_count >= max_frames:
                break

        idx += 1

    cap.release()

    if len(frames_out) == 0:
        raise ValueError(f"No frames selected from mask video: {video_path}")

    video_mask = torch.stack(frames_out).to(device=device, dtype=dtype)
    return video_mask


def min_bounding_rect(img):
    # print(img.dtype, img.shape)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print('Bad contours, using fake bbox...')
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # sort
    x_sorted = sorted(box, key=lambda x: x[0])
    left = x_sorted[:2]
    right = x_sorted[2:]
    left = sorted(left, key=lambda x: x[1])
    (tl, bl) = left
    right = sorted(right, key=lambda x: x[1])
    (tr, br) = right
    if tl[1] > bl[1]:
        (tl, bl) = (bl, tl)
    if tr[1] > br[1]:
        (tr, br) = (br, tr)
    return np.array([tl, tr, br, bl])

def adjust_image(box, img):
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    width = max(np.linalg.norm(pts1[0]-pts1[1]), np.linalg.norm(pts1[2]-pts1[3]))
    height = max(np.linalg.norm(pts1[0]-pts1[3]), np.linalg.norm(pts1[1]-pts1[2]))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # get transform matrix
    M = get_sym_mat(pts1, pts2, estimate_scale=True)
    C, H, W = img.shape
    T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
    theta = torch.from_numpy(theta[:2, :]).unsqueeze(0).type(torch.float32).to(img.device)
    grid = F.affine_grid(theta, torch.Size([1, C, H, W]), align_corners=True).to(img.dtype)
    result = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    # crop
    result = result[:, :int(height), :int(width)]
    return result

def crop_image(src_img, mask):
    # import pdb
    # pdb.set_trace()
    box = min_bounding_rect(mask)
    result = adjust_image(box, src_img)
    if len(result.shape) == 2:
        result = torch.stack([result]*3, axis=-1)
    return result

def crop_by_mask(img, mask):
    if mask.ndim == 3:
        mask = mask[0]

    ys, xs = torch.where(mask > 0)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return img[:, y1:y2+1, x1:x2+1]

class OCRLoss:
    def __init__(
        self,
        rec_model_dir,
        rec_char_dict_path,
        device,
        dtype,
        loss_alpha = 1,
        loss_beta = 1,
        latin_weight = 1,
        loss_type = 'l2',
    ):

        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.latin_weight = latin_weight
        self.device = device
        self.dtype = dtype

        text_predictor = create_predictor(rec_model_dir).to(device, dtype=dtype).eval()
        # text_predictor = create_predictor(rec_model_dir).to(device).eval()
        args_ocr = edict()
        args_ocr.rec_image_shape = "3, 96, 608"
        args_ocr.rec_batch_num = 6
        args_ocr.rec_char_dict_path = rec_char_dict_path
        args_ocr.use_fp16 = False
        self.cn_recognizer = TextRecognizer(args_ocr, text_predictor)
        # self.cn_recognizer.predictor.to(device).to(dtype)
        self.cn_recognizer.predictor.to(device)
        for param in text_predictor.parameters():
                param.requires_grad = False

        self.loss_type = loss_type

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def loss(self, image_pred, imgs, batch):
        bsz = image_pred.shape[0]
        bs_ocr_loss = []
        bs_ctc_loss = []

        lang_weight = []
        gt_texts = []
        x0_texts = []
        x0_texts_ori = []

        for i in range(bsz):
            n_lines = batch['n_lines'][i]  # batch size
            for j in range(n_lines):  # line
                lang = batch['language'][j][i]
                if lang == 'Chinese':
                    lang_weight += [1.0]
                elif lang == 'Latin':
                    lang_weight += [self.latin_weight]
                else:
                    lang_weight += [1.0]  # unsupport language, TODO
                gt_texts += [batch['texts'][j][i]]
                pos = batch['positions'][j][i]*255.
                # import pdb
                # pdb.set_trace()
                # pos = rearrange(pos, 'c h w -> h w c')
                np_pos = pos.detach().cpu().numpy().astype(np.uint8)
                x0_text = crop_image(image_pred[i], np_pos)
                x0_texts += [x0_text]
                x0_text_ori = crop_image(imgs[i], np_pos)
                x0_texts_ori += [x0_text_ori]

        if len(x0_texts) > 0:
            x0_list = x0_texts + x0_texts_ori
            x0_list = [x.to(imgs.dtype) for x in x0_list]
            # import pdb
            # pdb.set_trace()
            # preds shape: torch.Size([len(x0_list), 40, 6625])
            # preds_neck shape: torch.Size([len(x0_list), 40, 64])
            preds, preds_neck = self.cn_recognizer.pred_imglist(x0_list, show_debug=PRINT_DEBUG, norm=False)
            n_pairs = len(preds)//2
            preds_decode = preds[:n_pairs]  # preds_decode shape: torch.Size([len(x0_list) // 2, 40, 6625])
            preds_ori = preds[n_pairs:]     # preds_ori shape: torch.Size([len(x0_list) // 2, 40, 6625])
            preds_neck_decode = preds_neck[:n_pairs]    # preds_neck_decode shape: torch.Size([len(x0_list) // 2, 40, 64])
            preds_neck_ori = preds_neck[n_pairs:]   # preds_neck_ori shape: torch.Size([len(x0_list) // 2, 40, 64])
            lang_weight = torch.tensor(lang_weight).to(preds_neck.device)   # lang_weight shape: torch.Size([len(x0_list) // 2])
            # split to batches
            bs_preds_decode = []
            bs_preds_ori = []
            bs_preds_neck_decode = []
            bs_preds_neck_ori = []
            bs_lang_weight = []
            bs_gt_texts = []
            n_idx = 0
            for i in range(bsz):  # sample index in a batch
                n_lines = batch['n_lines'][i]
                bs_preds_decode += [preds_decode[n_idx:n_idx+n_lines]]
                bs_preds_ori += [preds_ori[n_idx:n_idx+n_lines]]
                bs_preds_neck_decode += [preds_neck_decode[n_idx:n_idx+n_lines]]
                bs_preds_neck_ori += [preds_neck_ori[n_idx:n_idx+n_lines]]
                bs_lang_weight += [lang_weight[n_idx:n_idx+n_lines]]
                bs_gt_texts += [gt_texts[n_idx:n_idx+n_lines]]
                n_idx += n_lines
            # calc loss
            ocr_loss_debug = []
            ctc_loss_debug = []

            for i in range(bsz):
                if len(bs_preds_neck_decode[i]) > 0:
                    if self.loss_alpha > 0:
                        sp_ocr_loss = self.get_loss(bs_preds_neck_decode[i], bs_preds_neck_ori[i], mean=False).mean([1, 2])
                        sp_ocr_loss *= bs_lang_weight[i]  # weighted by language
                        bs_ocr_loss += [sp_ocr_loss.mean()]
                        ocr_loss_debug += sp_ocr_loss.to(torch.float32).detach().cpu().numpy().tolist()
                    else:
                        bs_ocr_loss += [torch.tensor(0).float().to(pred_x0.device)]
                    if self.loss_beta > 0:
                        bs_preds_decode[i] = bs_preds_decode[i].to(torch.float32)
                        bs_lang_weight[i] = bs_lang_weight[i].to(torch.float32)
                        sp_ctc_loss = self.cn_recognizer.get_ctcloss(bs_preds_decode[i], bs_gt_texts[i], bs_lang_weight[i])
                        bs_ctc_loss += [sp_ctc_loss.mean()]
                        ctc_loss_debug += sp_ctc_loss.to(torch.float32).detach().cpu().numpy().tolist()
                    else:
                        bs_ctc_loss += [torch.tensor(0).float().to(pred_x0.device)]
                else:
                    bs_ocr_loss += [torch.tensor(0).float().to(pred_x0.device)]
                    bs_ctc_loss += [torch.tensor(0).float().to(pred_x0.device)]

            if PRINT_DEBUG and len(preds_decode) > 0:
                with torch.no_grad():
                    preds_all = preds_decode.softmax(dim=2)
                    preds_all_ori = preds_ori.softmax(dim=2)
                    for k in range(len(preds_all)):
                        pred = preds_all[k].to(torch.float32)
                        order, idx = self.cn_recognizer.decode(pred)
                        text = self.cn_recognizer.get_text(order)
                        pred_ori = preds_all_ori[k].to(torch.float32)
                        order, idx = self.cn_recognizer.decode(pred_ori)
                        text_ori = self.cn_recognizer.get_text(order)
                        str_log = f't = {t}, pred/ori/gt="{text}"/"{text_ori}"/"{gt_texts[k]}"'
                        if self.loss_alpha > 0:
                            str_log += f' ocr_loss={ocr_loss_debug[k]:.4f}'
                        if self.loss_beta > 0:
                            str_log += f' ctc_loss={ctc_loss_debug[k]:.4f}'
                        print(str_log)

            # loss_ocr += torch.stack(bs_ocr_loss) * self.loss_alpha * step_weight
            # loss_ctc += torch.stack(bs_ctc_loss) * self.loss_beta * step_weight
            # import pdb
            # pdb.set_trace()
            # loss_ocr += bs_ocr_loss[0] * self.loss_alpha
            # loss_ctc += bs_ctc_loss[0] * self.loss_beta
        
        step_weight = 1.0
        loss_ocr = torch.stack(bs_ocr_loss) * self.loss_alpha * step_weight
        loss_ctc = torch.stack(bs_ctc_loss) * self.loss_beta * step_weight

        res = {}
        res["loss_ocr"] = loss_ocr.mean()
        res["loss_ctc"] = loss_ctc.mean() 
        return res
    

    def new_loss(self, image_pred, imgs, image_masks, gt_texts, return_dict=False):
        """
        image_pred: (B, C, H, W)
        imgs:       (B, C, H, W)
        image_masks:(B, 1, H, W)
        gt_texts:   List[str], len = B
        """
        bsz = image_pred.shape[0]
        device = image_pred.device

        x0_texts = []
        x0_texts_ori = []
        valid_texts = []
        valid_ids = []

        # ---------- crop & collect ----------
        for i in range(bsz):
            x0 = crop_by_mask(image_pred[i], image_masks[i])
            x0_ori = crop_by_mask(imgs[i], image_masks[i])

            if x0 is None or x0_ori is None:
                continue

            # print(f"[{i}] x0:", x0.shape, "x0_ori:", x0_ori.shape)

            # save_tensor_image(x0, f'./debug/x0_{i}.png')
            # save_tensor_image(x0_ori, f'./debug/x0_ori_{i}.png')

            x0_texts.append(x0)
            x0_texts_ori.append(x0_ori)
            valid_texts.append(gt_texts[i])
            valid_ids.append(i)

        if len(x0_texts) == 0:
            zero = torch.tensor(0., device=device)
            return {"loss_ocr": zero, "loss_ctc": zero}

        # ---------- OCR forward ----------
        x0_list = x0_texts + x0_texts_ori
        x0_list = [x.to(imgs.dtype) for x in x0_list]

        preds, preds_neck = self.cn_recognizer.pred_imglist(
            x0_list, show_debug=False, norm=False
        )

        n = len(x0_texts)
        preds_decode = preds[:n]              # for CTC / decode
        preds_ori = preds[n:]                 # for debug
        preds_neck_decode = preds_neck[:n]    # pred image
        preds_neck_ori = preds_neck[n:]       # gt image

        # ---------- OCR consistency loss ----------
        sp_ocr_loss = self.get_loss(
            preds_neck_decode,
            preds_neck_ori,
            mean=False
        ).mean([1, 2])   # (n,)

        loss_ocr = sp_ocr_loss.mean() * self.loss_alpha

        # ---------- CTC loss ----------
        if self.loss_beta > 0:
            lang_weight = torch.ones(n, device=device)
            sp_ctc_loss = self.cn_recognizer.get_ctcloss(
                preds_decode.to(torch.float32),
                valid_texts,
                lang_weight
            )  # (n,)
            loss_ctc = sp_ctc_loss.mean() * self.loss_beta
        else:
            sp_ctc_loss = torch.zeros(n, device=device)
            loss_ctc = torch.tensor(0., device=device)

        # ---------- DEBUG PRINT ----------
        if PRINT_DEBUG:
            with torch.no_grad():
                preds_all = preds_decode.softmax(dim=2)
                preds_all_ori = preds_ori.softmax(dim=2)

                for k in range(n):
                    pred = preds_all[k].to(torch.float32)
                    order, _ = self.cn_recognizer.decode(pred)
                    text = self.cn_recognizer.get_text(order)

                    pred_ori = preds_all_ori[k].to(torch.float32)
                    order_ori, _ = self.cn_recognizer.decode(pred_ori)
                    text_ori = self.cn_recognizer.get_text(order_ori)

                    log = (
                        f"[sample {valid_ids[k]}] "
                        f'pred / ori / gt = "{text}" / "{text_ori}" / "{valid_texts[k]}"'
                    )

                    if self.loss_alpha > 0:
                        log += f" | ocr_loss={sp_ocr_loss[k].item():.4f}"
                    if self.loss_beta > 0:
                        log += f" | ctc_loss={sp_ctc_loss[k].item():.4f}"

                    print(log)

        if return_dict:
            return {
                "loss_ocr": loss_ocr,
                "loss_ctc": loss_ctc
            }
        else:
            return loss_ocr + loss_ctc
    

    def video_ocr_loss(
        self,
        image_pred,          # [3, T_gt, H, W]
        imgs,                # [3, T_gt, H, W]
        image_masks,         # [T_gt, 1, H, W]
        gt_texts,            # string
        num_supervise_frames=4,
        return_dict=False
    ):
        device = image_pred.device
        self.cn_recognizer.predictor.to(device)

        T = image_pred.shape[1]
        if T <= 0:
            zero = torch.tensor(0., device=device)
            if return_dict:
                return {"loss_ocr": zero, "loss_ctc": zero, "num_supervised_frames": 0}
            return zero, zero

        if num_supervise_frames == -1:
            n = T
        else:
            n = max(1, min(int(num_supervise_frames), T))
        
        idx = torch.linspace(0, T - 1, steps=n).long()

        image_pred = image_pred[:, idx]     # [3, n, H, W]
        imgs = imgs[:, idx]                 # [3, n, H, W]
        image_masks = image_masks[idx]      # [n, 1, H, W]
        if isinstance(gt_texts, (list, tuple)):
            gt_texts = gt_texts[0] if len(gt_texts) > 0 else ""
        gt_texts = str(gt_texts)

        x0_texts = []
        x0_texts_ori = []

        for i in range(n):
            pred_frame = image_pred[:, i]   # [3,H,W]
            gt_frame = imgs[:, i]
            mask = image_masks[i]

            x0 = crop_by_mask(pred_frame, mask)
            x0_ori = crop_by_mask(gt_frame, mask)

            if x0 is None or x0_ori is None:
                continue

            x0_texts.append(x0)
            x0_texts_ori.append(x0_ori)

        if len(x0_texts) == 0:
            zero = torch.tensor(0., device=device)
            if return_dict:
                return {
                    "loss_ocr": zero,
                    "loss_ctc": zero,
                    "num_supervised_frames": 0,
                }
            else:
                return zero, zero

        x0_list = x0_texts + x0_texts_ori
        x0_list = [x.to(device=device, dtype=image_pred.dtype) for x in x0_list]

        preds, preds_neck = self.cn_recognizer.pred_imglist(
            x0_list,
            show_debug=False,
            norm=False
        )

        k = len(x0_texts)
        preds_decode = preds[:k]
        preds_neck_decode = preds_neck[:k]
        preds_neck_ori = preds_neck[k:]

        sp_ocr_loss = self.get_loss(
            preds_neck_decode,
            preds_neck_ori,
            mean=False
        ).mean([1, 2])

        loss_ocr = sp_ocr_loss.mean() * self.loss_alpha

        if self.loss_beta > 0:
            lang_weight = torch.ones(k, device=device)
            texts = [gt_texts] * k
            sp_ctc_loss = self.cn_recognizer.get_ctcloss(
                preds_decode.to(torch.float32),
                texts,
                lang_weight
            )
            loss_ctc = sp_ctc_loss.mean() * self.loss_beta
        else:
            sp_ctc_loss = torch.zeros(k, device=device)
            loss_ctc = torch.tensor(0., device=device)
        
        if PRINT_DEBUG:
            with torch.no_grad():
                preds_all = preds_decode.softmax(dim=2)
                preds_all_ori = preds[k:].softmax(dim=2)

                for i in range(k-1, k):
                    pred = preds_all[i].to(torch.float32)
                    order, _ = self.cn_recognizer.decode(pred)
                    text = self.cn_recognizer.get_text(order)

                    pred_ori = preds_all_ori[i].to(torch.float32)
                    order_ori, _ = self.cn_recognizer.decode(pred_ori)
                    text_ori = self.cn_recognizer.get_text(order_ori)

                    log = (
                        f"[video frame {idx[i].item()}] "
                        f'pred / ori / gt = "{text}" / "{text_ori}" / "{gt_texts}"'
                    )

                    if self.loss_alpha > 0:
                        log += f" | ocr_loss={sp_ocr_loss[i].item():.4f}"
                    if self.loss_beta > 0:
                        log += f" | ctc_loss={sp_ctc_loss[i].item():.4f}"
                    print(log)

        if return_dict:
            return {
                "loss_ocr": loss_ocr,
                "loss_ctc": loss_ctc,
                "num_supervised_frames": k,
            }
        else:
            return loss_ocr, loss_ctc


if __name__ == '__main__':
    ocr_loss_config = {
        "rec_model_dir": str(DEFAULT_OCR_MODEL_DIR),
        "rec_char_dict_path": str(DEFAULT_OCR_CHAR_DICT_PATH),
        "device": "cuda",
        "dtype": torch.bfloat16,
    }
    ocr_loss = OCRLoss(**ocr_loss_config)

    # test
    video_path = "data/text_editing/all_mixkit_easyadd/Airplane/mixkit-a-smartphone-records-a-young-woman-posing-in-a-room-50486/65-130/clip1.mp4"
    video_pred_path = "data/text_editing/all_mixkit_easyadd/Airplane/mixkit-a-smartphone-records-a-young-woman-posing-in-a-room-50486/65-130/clip2.mp4"
    video_mask_path = "data/text_editing/all_mixkit_easyadd/Airplane/mixkit-a-smartphone-records-a-young-woman-posing-in-a-room-50486/65-130/mask.mp4"

    video_preds = read_video_cv2(video_pred_path, device=ocr_loss.device, dtype=ocr_loss.dtype, frame_stride=4)
    video_imgs = read_video_cv2(video_path, device=ocr_loss.device, dtype=ocr_loss.dtype, frame_stride=4)
    video_masks = read_mask_video_cv2(video_mask_path, device=ocr_loss.device, dtype=ocr_loss.dtype, frame_stride=4)
    # print(video_preds.shape, video_imgs.shape, video_masks.shape)

    gt_text = 'boomers beacon decayed'

    res = ocr_loss.video_ocr_loss(video_preds, video_imgs, video_masks, gt_text, num_supervise_frames=-1)
    print(res)
