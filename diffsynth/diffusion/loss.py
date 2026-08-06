from .base_pipeline import BasePipeline
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def _pil_mask_to_tensor(mask_image: Image.Image) -> torch.Tensor:
    mask_np = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(mask_np)


def _normalize_video_mask(mask_input, device, target_shape) -> torch.Tensor | None:
    if mask_input is None:
        return None

    mask = None
    if torch.is_tensor(mask_input):
        mask = mask_input
    elif isinstance(mask_input, Image.Image):
        mask = _pil_mask_to_tensor(mask_input)  # [H, W]
    elif isinstance(mask_input, (list, tuple)):
        if len(mask_input) == 0:
            return None
        first = mask_input[0]
        if isinstance(first, Image.Image):
            mask = torch.stack([_pil_mask_to_tensor(img) for img in mask_input], dim=0)  # [T, H, W]
        elif torch.is_tensor(first):
            mask = torch.stack(list(mask_input), dim=0)
        elif isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], Image.Image):
            video_masks = []
            for video_mask in mask_input:
                if len(video_mask) == 0:
                    continue
                video_masks.append(torch.stack([_pil_mask_to_tensor(img) for img in video_mask], dim=0))
            if len(video_masks) == 0:
                return None
            mask = torch.stack(video_masks, dim=0)  # [B, T, H, W]

    if mask is None:
        return None

    mask = mask.to(device=device, dtype=torch.float32)

    # To [B, 1, T, H, W]
    if mask.ndim == 2:  # [H, W]
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:  # [T, H, W]
        mask = mask.unsqueeze(0).unsqueeze(1)
    elif mask.ndim == 4:
        target_b = target_shape[0]
        # [B, T, H, W]
        if mask.shape[0] == target_b and mask.shape[1] > 4:
            mask = mask.unsqueeze(1)
        # [T, C, H, W]
        elif mask.shape[1] in (1, 3):
            mask = mask[:, :1].permute(1, 0, 2, 3).unsqueeze(0)
        # [C, T, H, W]
        elif mask.shape[0] in (1, 3) and mask.shape[1] > 4:
            mask = mask[:1].unsqueeze(0)
        # fallback as [B, T, H, W]
        else:
            mask = mask.unsqueeze(1)
    elif mask.ndim == 5:
        # [B, C, T, H, W]
        if mask.shape[1] in (1, 3):
            mask = mask[:, :1]
        # [B, T, C, H, W]
        elif mask.shape[2] in (1, 3):
            mask = mask[:, :, :1].permute(0, 2, 1, 3, 4)
        else:
            return None
    else:
        return None

    # Binarize and resize to latent shape.
    mask = (mask > 0.5).to(dtype=torch.float32)
    target_b, _, target_t, target_h, target_w = target_shape
    mask = F.interpolate(mask, size=(target_t, target_h, target_w), mode="nearest")

    if mask.shape[0] != target_b:
        if mask.shape[0] == 1:
            mask = mask.expand(target_b, -1, -1, -1, -1)
        elif target_b % mask.shape[0] == 0:
            mask = mask.repeat(target_b // mask.shape[0], 1, 1, 1, 1)
        else:
            mask = mask[:1].expand(target_b, -1, -1, -1, -1)

    return mask


def _normalize_gt_text(gt_text) -> str:
    if gt_text is None:
        return ""
    if isinstance(gt_text, str):
        return gt_text
    if isinstance(gt_text, (list, tuple)):
        if len(gt_text) == 0:
            return ""
        return "" if gt_text[0] is None else str(gt_text[0])
    if isinstance(gt_text, float) and np.isnan(gt_text):
        return ""
    return str(gt_text)


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    per_pixel_mse = torch.nn.functional.mse_loss(
        noise_pred.float(),
        training_target.float(),
        reduction="none",
    )
    naive_loss = per_pixel_mse.mean()
    reweight_loss = naive_loss
    mask_ratio = torch.tensor(0.0, device=pipe.device)
    mask_weight = torch.tensor(1.0, device=pipe.device)
    loss_ocr = torch.tensor(0.0, device=pipe.device)
    loss_ctc = torch.tensor(0.0, device=pipe.device)

    mask_reweight_loss_scale = float(inputs.get("mask_reweight_loss_scale", 0.0))
    if mask_reweight_loss_scale > 0:
        mask_map = _normalize_video_mask(
            inputs.get("vace_video_mask"),
            device=pipe.device,
            target_shape=per_pixel_mse.shape,
        )
        if mask_map is not None:
            # Foreground weight: 1 + fixed scale. Background remains 1.
            weight_map = 1.0 + mask_map * mask_reweight_loss_scale
            # reweight_loss = (per_pixel_mse * weight_map).mean()
            weighted = per_pixel_mse * weight_map
            reweight_loss = weighted.sum() / weight_map.sum()
            mask_ratio = mask_map.mean()
            mask_weight = torch.tensor(1.0 + mask_reweight_loss_scale, device=pipe.device)
        else:
            reweight_loss = per_pixel_mse.mean()
    else:
        reweight_loss = per_pixel_mse.mean()

    loss = reweight_loss

    ocr_loss_scale = float(inputs.get("ocr_loss_scale", 0.0))
    if getattr(pipe, "ocr_loss", None) is not None:
        ocr_use_grad = ocr_loss_scale > 0
        gt_text = _normalize_gt_text(inputs.get("gt_text"))
        vace_video_mask = inputs.get("vace_video_mask")
        if len(gt_text) > 0 and vace_video_mask is not None:
            if hasattr(pipe, "load_models_to_device"):
                try:
                    pipe.load_models_to_device(["vae"])
                except Exception:
                    pass

            sigma = pipe.scheduler.sigmas[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
            latents_for_ocr = inputs["latents"]
            if latents_for_ocr.shape[2] != noise_pred.shape[2]:
                if latents_for_ocr.shape[2] == noise_pred.shape[2] + 1:
                    latents_for_ocr = latents_for_ocr[:, :, 1:]
                else:
                    latents_for_ocr = latents_for_ocr[:, :, :noise_pred.shape[2]]
            x0_est = latents_for_ocr - sigma * noise_pred
            target_latents_for_ocr = inputs["input_latents"]
            if target_latents_for_ocr.shape[2] != x0_est.shape[2]:
                if target_latents_for_ocr.shape[2] == x0_est.shape[2] + 1:
                    target_latents_for_ocr = target_latents_for_ocr[:, :, 1:]
                else:
                    target_latents_for_ocr = target_latents_for_ocr[:, :, :x0_est.shape[2]]

            ref_frames = 0
            vace_reference_image = inputs.get("vace_reference_image")
            if vace_reference_image is not None:
                ref_frames = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            if ref_frames > 0 and x0_est.shape[2] > ref_frames and target_latents_for_ocr.shape[2] > ref_frames:
                x0_est = x0_est[:, :, ref_frames:]
                target_latents_for_ocr = target_latents_for_ocr[:, :, ref_frames:]

            common_t = min(x0_est.shape[2], target_latents_for_ocr.shape[2])
            if common_t > 0:
                x0_est = x0_est[:, :, :common_t]
                target_latents_for_ocr = target_latents_for_ocr[:, :, :common_t]

            if x0_est.shape[2] > 0 and target_latents_for_ocr.shape[2] > 0:
                # Sample frames on latent timeline before VAE decode to reduce OCR-branch memory.
                num_supervise_frames = int(inputs.get("ocr_num_supervise_frames", 1))
                if num_supervise_frames > 0 and x0_est.shape[2] > num_supervise_frames:
                    frame_idx = torch.linspace(
                        0,
                        x0_est.shape[2] - 1,
                        steps=num_supervise_frames,
                        device=x0_est.device,
                    ).long()
                    x0_est = x0_est.index_select(2, frame_idx)
                    target_latents_for_ocr = target_latents_for_ocr.index_select(2, frame_idx)

                if ocr_use_grad:
                    pred_images = pipe.vae.decode(x0_est, device=pipe.device)
                else:
                    with torch.no_grad():
                        pred_images = pipe.vae.decode(x0_est, device=pipe.device)
                with torch.no_grad():
                    target_images = pipe.vae.decode(target_latents_for_ocr, device=pipe.device)

                mask_map = _normalize_video_mask(
                    vace_video_mask,
                    device=pipe.device,
                    target_shape=(pred_images.shape[0], 1, pred_images.shape[2], pred_images.shape[3], pred_images.shape[4]),
                )
                if mask_map is not None and pred_images.shape[0] > 0:
                    video_mask = mask_map[0].permute(1, 0, 2, 3).contiguous()
                    loss_ocr, loss_ctc = pipe.ocr_loss.video_ocr_loss(
                        pred_images[0],
                        target_images[0],
                        video_mask,
                        gt_text,
                        num_supervise_frames=-1,
                        return_dict=False,
                    )
                    if ocr_use_grad:
                        loss = loss + (loss_ocr + loss_ctc) * ocr_loss_scale

    final_loss = loss * pipe.scheduler.training_weight(timestep)
    loss_dict = {
        "loss_total": final_loss.detach().item(),
        "loss_reweight": reweight_loss.detach().item(),
        "loss_naive": naive_loss.detach().item(),
        "mask_ratio": mask_ratio.detach().item(),
        "mask_weight": mask_weight.detach().item(),
        "loss_ocr": loss_ocr.detach().item(),
        "loss_ctc": loss_ctc.detach().item(),
    }
    return final_loss, loss_dict

# def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
#     max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
#     min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

#     timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
#     timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
#     noise = torch.randn_like(inputs["input_latents"])
#     inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
#     training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
#     if "first_frame_latents" in inputs:
#         inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
#     models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
#     noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
#     if "first_frame_latents" in inputs:
#         noise_pred = noise_pred[:, :, 1:]
#         training_target = training_target[:, :, 1:]
    
#     loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
#     loss = loss * pipe.scheduler.training_weight(timestep)
#     return loss

def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
