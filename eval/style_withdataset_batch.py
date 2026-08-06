import os
import piq
import math
import torch
import argparse
import warnings
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
from scipy import signal, ndimage
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as skimage_ssim

from style_utils import devdata, fspecial_gauss
warnings.filterwarnings("ignore", message="nn.functional.upsample is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    size = min(img1.shape[0], 11)
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003

    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.convolve(img1, downsample_filter, mode='reflect')
        filtered_im2 = ndimage.convolve(img2, downsample_filter, mode='reflect')
        im1 = filtered_im1[:: 2, :: 2]
        im2 = filtered_im2[:: 2, :: 2]
    # Note: Remove the negative and add it later to avoid NaN in exponential.
    sign_mcs = np.sign(mcs[0: level - 1])
    sign_mssim = np.sign(mssim[level - 1])
    mcs_power = np.power(np.abs(mcs[0: level - 1]), weight[0: level - 1])
    mssim_power = np.power(np.abs(mssim[level - 1]), weight[level - 1])
    return np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power

def compute_cided2000(img1, img2, normalize=False, mask=None, delta_e_max=50):
    """
    Compute the CIEDE2000 color difference between two images.
    If normalize=True, return similarity in [0,1] via 1 - (mean_deltaE / delta_e_max).
    """
    import numpy as np
    from skimage.color import rgb2lab, deltaE_ciede2000
    import torch

    # Convert tensors to numpy arrays
    def to_numpy(img):
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        return img

    arr1 = to_numpy(img1)
    arr2 = to_numpy(img2)
    # Squeeze batch and reorder to HWC
    if arr1.ndim == 4:
        arr1 = arr1.squeeze(0)
    if arr2.ndim == 4:
        arr2 = arr2.squeeze(0)
    # Channel first to last
    if arr1.shape[0] == 3:
        arr1 = np.transpose(arr1, (1, 2, 0))
    if arr2.shape[0] == 3:
        arr2 = np.transpose(arr2, (1, 2, 0))
    # Scale [0,1] to [0,255]
    if arr1.max() <= 1.0:
        arr1 = (arr1 * 255).astype(np.uint8)
    else:
        arr1 = arr1.astype(np.uint8)
    if arr2.max() <= 1.0:
        arr2 = (arr2 * 255).astype(np.uint8)
    else:
        arr2 = arr2.astype(np.uint8)

    # Convert RGB to Lab
    lab1 = rgb2lab(arr1)
    lab2 = rgb2lab(arr2)
    # Compute deltaE per pixel
    delta = deltaE_ciede2000(lab1, lab2)
    # Apply mask if provided
    if mask is not None:
        delta = delta[mask]
    # Mean deltaE
    mean_delta = float(np.mean(delta))
    if normalize:
        sim = 1.0 - min(mean_delta / delta_e_max, 1.0)
        return sim
    return mean_delta

def compute_fsim(img1, img2, device: torch.device):
    """
    Compute Feature Similarity Index (FSIM) for two images.
    """
    # Convert numpy to tensor
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    
    # Ensure batch dimension
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    # Move to device and float
    img1 = img1.to(device).float()
    img2 = img2.to(device).float()
    # Clamp values to [0,1] to satisfy piq.fsim requirements
    img1 = img1.clamp(0.0, 1.0)
    img2 = img2.clamp(0.0, 1.0)
    
    # Compute FSIM using piq library
    value = piq.fsim(img1, img2, data_range=1.0)
    
    # Normalize if required
    return value.item()

def compute_msssim(img1, img2, levels=5, weights=None):
    if weights is None:
        weights = np.array([0.0448,0.2856,0.3001,0.2363,0.1333])
    
    # Convert tensors to numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Squeeze batch dimension if present
    if img1.ndim == 4:
        img1 = img1.squeeze(0)
    if img2.ndim == 4:
        img2 = img2.squeeze(0)
    
    # Channel first to last
    if img1.shape[0] in (1, 3):
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] in (1, 3):
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Ensure float32 type
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Check minimum size requirement
    min_size = 11  # minimum window size for SSIM
    if img1.shape[0] < min_size or img1.shape[1] < min_size:
        # If image is too small, return fallback SSIM
        try:
            return float(skimage_ssim(img1, img2, data_range=1.0, channel_axis=-1))
        except:
            return 0.0
    
    mssim_vals, mcs_vals = [], []

    for l in range(levels):
        try:
            # Check if image is still large enough
            if img1.shape[0] < min_size or img1.shape[1] < min_size:
                break
                
            ssim_map, cs_map = skimage_ssim(img1, img2, full=True, data_range=1.0, channel_axis=-1)
            
            # Check for NaN or invalid values
            if np.isnan(ssim_map).any() or np.isnan(cs_map).any():
                break
                
            mssim_vals.append(float(ssim_map.mean()))
            mcs_vals.append(float(cs_map.mean()))

            # Gaussian filter + downsample
            img1 = gaussian_filter(img1, sigma=1)[::2, ::2]
            img2 = gaussian_filter(img2, sigma=1)[::2, ::2]
            
        except Exception as e:
            # If any error occurs, break the loop
            break

    # Check if we have enough valid measurements
    if len(mssim_vals) == 0:
        return 0.0
    
    # Adjust levels and weights based on actual number of measurements
    actual_levels = len(mssim_vals)
    if actual_levels < levels:
        weights = weights[:actual_levels]
        weights = weights / weights.sum()  # renormalize
    
    mssim_vals = np.array(mssim_vals)
    mcs_vals = np.array(mcs_vals)

    # Check for any NaN or invalid values
    if np.isnan(mssim_vals).any() or np.isnan(mcs_vals).any():
        return 0.0
    
    # Ensure all values are positive for power calculation
    mcs_vals = np.abs(mcs_vals)
    mssim_vals = np.abs(mssim_vals)
    
    # Add small epsilon to avoid zero values
    epsilon = 1e-10
    mcs_vals = np.maximum(mcs_vals, epsilon)
    mssim_vals = np.maximum(mssim_vals, epsilon)

    try:
        if actual_levels == 1:
            overall = mssim_vals[0]
        else:
            overall = np.prod(mcs_vals[:-1]**weights[:-1]) * (mssim_vals[-1]**weights[-1])
        
        # Check if result is valid
        if np.isnan(overall) or np.isinf(overall):
            return 0.0
        
        return float(overall)
        
    except Exception as e:
        return 0.0


def compute_text_appearance_similarity(sim_color, sim_font, sim_removal):

    text_style_similarity = (sim_color + sim_font + sim_removal) / 3.0
    
    return text_style_similarity


def calculate_metrics_for_images(img_tensor, gt_tensor, device: torch.device, img_path='', gt_path=''):
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    if gt_tensor.dim() == 3:
        gt_tensor = gt_tensor.unsqueeze(0)
    
    # MSE
    mse = ((gt_tensor - img_tensor) ** 2).mean().item()
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10(1.0 / mse)
    
    # SSIM
    R = gt_tensor[0, 0, :, :]
    G = gt_tensor[0, 1, :, :]
    B = gt_tensor[0, 2, :, :]
    YGT = .299 * R + .587 * G + .114 * B
    
    R = img_tensor[0, 0, :, :]
    G = img_tensor[0, 1, :, :]
    B = img_tensor[0, 2, :, :]
    YBC = .299 * R + .587 * G + .114 * B
    
    ssim_value = msssim(np.array(YGT * 255), np.array(YBC * 255))

    # No StyleNet inference: directly use the original src/gt images for
    # color, font and removal related similarity metrics.
    cided2000_color_norm = compute_cided2000(
        img_tensor, gt_tensor, normalize=True, mask=None, delta_e_max=50
    )
    fsim_font = compute_fsim(img_tensor, gt_tensor, device=device)
    msssim_removal = compute_msssim(img_tensor, gt_tensor)

    tas_score = compute_text_appearance_similarity(cided2000_color_norm, fsim_font, msssim_removal)
    
    metrics = {
        'ssim': float(ssim_value),
        'psnr': float(psnr),
        'mse': float(mse),
        's_color': float(cided2000_color_norm),
        's_font': float(fsim_font),
        's_bg': float(msssim_removal),
        'tas_score': float(tas_score),
        'img_path': img_path,
        'gt_path': gt_path,
    }
    
    return metrics


@dataclass
class EvalJob:
    target_path: Path
    gt_path: Path
    output_txt: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch style evaluation.")
    parser.add_argument("--target_path", type=str, default=None, help="Path to predicted images for a single job.")
    parser.add_argument("--gt_path", type=str, default=None, help="Path to ground-truth images for a single job.")
    parser.add_argument("--output_txt", type=str, default=None, help="Output txt path for a single job.")
    parser.add_argument(
        "--job",
        nargs=3,
        action="append",
        metavar=("TARGET_PATH", "GT_PATH", "OUTPUT_TXT"),
        help="Batch job triple. Repeat this arg to evaluate multiple jobs in one process.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader num_workers.")
    parser.add_argument("--fid_batch_size", type=int, default=1, help="Batch size used by FID.")
    parser.add_argument("--fid_dims", type=int, default=2048, help="Feature dims used by FID.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for metric computation.")
    return parser.parse_args()


def collect_jobs(args: argparse.Namespace) -> List[EvalJob]:
    jobs: List[EvalJob] = []

    if args.job:
        for target_path_str, gt_path_str, output_txt_str in args.job:
            jobs.append(
                EvalJob(
                    target_path=Path(target_path_str),
                    gt_path=Path(gt_path_str),
                    output_txt=Path(output_txt_str),
                )
            )

    if args.target_path and args.gt_path:
        output_txt = args.output_txt if args.output_txt else str(Path(args.target_path) / "style_eval.txt")
        jobs.append(
            EvalJob(
                target_path=Path(args.target_path),
                gt_path=Path(args.gt_path),
                output_txt=Path(output_txt),
            )
        )

    if not jobs:
        raise ValueError("Provide at least one of: --job, or both --target_path and --gt_path.")

    return jobs


def _format_metric(value):
    if value is None:
        return "nan"
    return f"{float(value):.4f}"


def write_avg_metrics(output_txt: Path, avg_metrics: Dict[str, float]) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f'SSIM(↑): {_format_metric(avg_metrics.get("ssim"))}\n')
        f.write(f'PSNR(↑): {_format_metric(avg_metrics.get("psnr"))}\n')
        f.write(f'MSE(↓): {_format_metric(avg_metrics.get("mse"))}\n')
        f.write(f'FID(↓): {_format_metric(avg_metrics.get("fid"))}\n')
        f.write(f'Color CIEDE2000 Norm(↑): {_format_metric(avg_metrics.get("s_color"))}\n')
        f.write(f'Font FSIM(↑): {_format_metric(avg_metrics.get("s_font"))}\n')
        f.write(f'Removal MSSSIM(↑): {_format_metric(avg_metrics.get("s_bg"))}\n')
        f.write(f'Text Style Similarity w/FSIM+MSSSIM(↑): {_format_metric(avg_metrics.get("tas_score"))}\n')


def eval_one_job(job: EvalJob, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    target_path = job.target_path
    gt_path = job.gt_path
    output_txt = job.output_txt

    if not target_path.exists():
        raise FileNotFoundError(f"target_path not found: {target_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"gt_path not found: {gt_path}")

    img_data = devdata(dataRoot=str(target_path), gtRoot=str(gt_path))
    data_loader = DataLoader(
        img_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    print(f"Found {len(img_data)} images to evaluate.")

    sum_metrics = {
        "ssim": 0.0,
        "psnr": 0.0,
        "mse": 0.0,
        "s_color": 0.0,
        "s_font": 0.0,
        "s_bg": 0.0,
        "tas_score": 0.0,
    }
    valid_count = 0
    skipped_count = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()

    for _, (img, lbl, path) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Style eval ({target_path.name})"):
        try:
            filename = path[0] if isinstance(path, (list, tuple)) else path
            img_file_path = os.path.join(str(target_path), filename)
            gt_file_path = os.path.join(str(gt_path), filename)
            metrics = calculate_metrics_for_images(img, lbl, device, img_file_path, gt_file_path)
        except Exception as e:
            skipped_count += 1
            print(f"Skip img {path} because of {e}")
            continue

        for key in sum_metrics.keys():
            sum_metrics[key] += metrics[key]
        valid_count += 1

    print("Calculating FID score...")
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [str(gt_path), str(target_path)],
            args.fid_batch_size,
            device,
            args.fid_dims,
        )
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid_value = None

    if valid_count == 0:
        raise RuntimeError("No valid image pairs were processed.")

    avg_metrics = {k: v / valid_count for k, v in sum_metrics.items()}
    avg_metrics["fid"] = fid_value
    write_avg_metrics(output_txt, avg_metrics)

    return {
        "total_images": len(img_data),
        "valid_images": valid_count,
        "skipped_images": skipped_count,
        **avg_metrics,
    }


def main() -> None:
    args = parse_args()
    jobs = collect_jobs(args)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if args.use_gpu and device.type != "cuda":
        print("[WARN] --use_gpu is set, but CUDA is unavailable. Falling back to CPU.")

    all_job_metrics: List[Dict[str, float]] = []
    for idx, job in enumerate(jobs):
        print("")
        print(f"[JOB {idx + 1}/{len(jobs)}] target_path={job.target_path}")
        print(f"[JOB {idx + 1}/{len(jobs)}] gt_path={job.gt_path}")
        print(f"[JOB {idx + 1}/{len(jobs)}] output_txt={job.output_txt}")
        try:
            metrics = eval_one_job(job=job, args=args, device=device)
            all_job_metrics.append(metrics)
            print(f"Average results saved to {job.output_txt}")
            print(f"Total images: {metrics['total_images']}")
            print(f"Valid images: {metrics['valid_images']}")
            print(f"Skipped images: {metrics['skipped_images']}")
            print(f"TAS score: {_format_metric(metrics['tas_score'])}")
            print(f"FID: {_format_metric(metrics['fid'])}")
        except Exception as e:
            print(f"[ERROR] Style eval job failed: {e}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not all_job_metrics:
        raise RuntimeError("No style eval jobs succeeded.")

    if len(all_job_metrics) > 1:
        total_images = sum(m["total_images"] for m in all_job_metrics)
        valid_images = sum(m["valid_images"] for m in all_job_metrics)
        skipped_images = sum(m["skipped_images"] for m in all_job_metrics)

        weighted = {}
        for key in ["ssim", "psnr", "mse", "s_color", "s_font", "s_bg", "tas_score"]:
            weighted[key] = sum(m[key] * m["valid_images"] for m in all_job_metrics) / max(valid_images, 1)

        fid_values = [m["fid"] for m in all_job_metrics if m["fid"] is not None]
        weighted["fid"] = (sum(fid_values) / len(fid_values)) if fid_values else None

        print("")
        print("========== Style Eval Summary ==========")
        print(f"jobs: {len(all_job_metrics)}")
        print(f"total_images: {total_images}")
        print(f"valid_images: {valid_images}")
        print(f"skipped_images: {skipped_images}")
        print(f"weighted_tas_score: {_format_metric(weighted['tas_score'])}")
        print(f"mean_fid: {_format_metric(weighted['fid'])}")
        print("========================================")


if __name__ == "__main__":
    main()
