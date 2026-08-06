import os
import numpy as np
import torch
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from scipy import linalg
from tqdm import tqdm
from decord import VideoReader, cpu

from pytorchvideo.models.hub import i3d_r50


# =============================
# config
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_LEN = 16
NUM_CLIPS = 1
BATCH_SIZE = 16

IMG_SIZE = 224
NUM_WORKERS = 16

clip_queue = queue.Queue(maxsize=1024)
map_queue = queue.Queue()


# =============================
# clip sampler
# =============================
def sample_clips(total_frames, clip_len=16, num_clips=4):
    if total_frames <= clip_len:
        return [np.arange(total_frames) for _ in range(num_clips)]

    max_start = total_frames - clip_len
    starts = np.linspace(0, max_start, num_clips).astype(int)

    return [np.arange(s, s + clip_len) for s in starts]


# =============================
# worker: decode video → clips
# =============================
def video_worker(path):
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)

    clips_idx = sample_clips(total, CLIP_LEN, NUM_CLIPS)

    local_clips = []

    for idx in clips_idx:
        frames = vr.get_batch(idx).asnumpy()

        frames = torch.tensor(frames).float() / 255.0

        frames = torch.nn.functional.interpolate(
            frames.permute(0, 3, 1, 2),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )

        local_clips.append(frames)

    start = clip_queue.qsize()

    for c in local_clips:
        clip_queue.put(c)

    map_queue.put((start, start + len(local_clips)))


# =============================
# multi-thread video loader
# =============================
def collect_multithread(folder):
    videos = [
        os.path.join(folder, v)
        for v in os.listdir(folder)
        if v.endswith((".mp4", ".avi", ".mov"))
    ]

    print(f"Decoding videos with {NUM_WORKERS} threads...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        list(tqdm(ex.map(video_worker, videos), total=len(videos)))


# =============================
# I3D model
# =============================
class I3DExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = i3d_r50(pretrained=True)
        self.model.blocks[5].proj = torch.nn.Identity()
        self.model = self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # B,C,T,H,W
        return self.model(x)


# =============================
# GPU consumer (true batch)
# =============================
def gpu_encode(model, total_clips):
    feats = []
    clip_list = []

    for _ in range(total_clips):
        clip_list.append(clip_queue.get())

    for i in tqdm(range(0, len(clip_list), BATCH_SIZE), desc="GPU encoding"):
        batch = clip_list[i:i + BATCH_SIZE]
        batch = torch.stack(batch).to(DEVICE)

        with torch.no_grad():
            f = model(batch).cpu().numpy()

        feats.append(f)

    return np.concatenate(feats, axis=0)


# =============================
# regroup per video
# =============================
def pool_video_features(all_feats, video_map):
    video_feats = []

    for start, end in video_map:
        video_feats.append(all_feats[start:end].mean(axis=0))

    return np.stack(video_feats)


# =============================
# feature extraction pipeline
# =============================
def extract_features(folder, model):
    clip_queue.queue.clear()
    map_queue.queue.clear()

    collect_multithread(folder)

    total_clips = clip_queue.qsize()

    print(f"Total clips: {total_clips}")

    clip_feats = gpu_encode(model, total_clips)

    video_map = []

    while not map_queue.empty():
        video_map.append(map_queue.get())

    video_feats = pool_video_features(clip_feats, video_map)

    return video_feats


# =============================
# FVD
# =============================
def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)


def compute_fvd(real_feats, fake_feats):
    mu1, mu2 = real_feats.mean(0), fake_feats.mean(0)
    sigma1 = np.cov(real_feats, rowvar=False)
    sigma2 = np.cov(fake_feats, rowvar=False)

    return frechet_distance(mu1, sigma1, mu2, sigma2)


# =============================
# main
# =============================
def run_fvd(real_dir, fake_dir):
    print("Loading I3D...")
    model = I3DExtractor()

    print("\n=== REAL ===")
    real_feats = extract_features(real_dir, model)

    print("\n=== FAKE ===")
    fake_feats = extract_features(fake_dir, model)

    print("\nComputing FVD...")
    return compute_fvd(real_feats, fake_feats)


# =============================
# entry
# =============================
if __name__ == "__main__":
    # real_dir = "output/infer_multi/vace_14b_stage2_v4_linear_qwenvlm_addocremb_prompt1_from_ogrloss/realbench_prompt1/step-1000"
    # fake_dir = "output/infer_multi/vace_14b_stage2_v4_linear_qwenvlm_addocremb_prompt1_from_ogrloss/realbench_prompt1/step-25168"
    # 1.20

    # real_dir = "output/vte_exp/viva/outputs/ema12000_w2/src"
    # fake_dir = "output/vte_exp/viva/outputs/ema12000_w2/tgt"
    # 3.06

    real_dir = "output/infer_multi/vace_14b_stage2_v4_linear_qwenvlm_addocremb_prompt1_from_ogrloss/realbench_prompt1/step-1000"
    fake_dir = "output/infer_multi/vace_14b_stage2_v4_linear_qwenvlm_addocremb_prompt1_from_ogrloss/synbench_prompt1/step-1000"
    # 88.35

    fvd = run_fvd(real_dir, fake_dir)

    print("\n🔥 FVD:", fvd)