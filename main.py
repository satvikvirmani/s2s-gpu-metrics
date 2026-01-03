from __future__ import annotations

import cv2
import json
import uuid
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Callable, Dict, List

import numpy as np
import torch
import lpips
from fastapi import FastAPI, WebSocket

from metrics import Metrics

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metrics_api")

# ---------------------------------------------------------------------
# GLOBAL DEVICE (single source of truth)
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BATCH_SIZE = 16
MIN_FRAMES_FOR_GPU = 120

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GPU_SEMAPHORE = asyncio.Semaphore(1)

# ---------------------------------------------------------------------
# GLOBAL MODELS
# ---------------------------------------------------------------------
lpips_model: lpips.LPIPS | None = None

# ---------------------------------------------------------------------
# APP LIFESPAN
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global lpips_model

    logger.info("Loading LPIPS model on %s", DEVICE)
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE).eval()

    yield

    logger.info("Application shutdown")

app = FastAPI(
    title="Skip2Smooth GPU Metrics API",
    lifespan=lifespan,
    root_path="/.lightning/proxy",
)

# ---------------------------------------------------------------------
# METRIC PIPELINE
# ---------------------------------------------------------------------
def metrics_wrapper(
    video_path: Path,
    progress_cb: Callable[[float, str], None] | None,
) -> np.ndarray:
    """
    Compute frame-wise metrics for a video.

    Returns:
        np.ndarray of shape (num_frames - 1, 4)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < MIN_FRAMES_FOR_GPU:
        raise ValueError("Video too small for GPU metric computation")

    metrics = np.zeros((total_frames - 1, 4), dtype=np.float32)

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Failed to read first frame")

    batch_t1: List[torch.Tensor] = []
    batch_t2: List[torch.Tensor] = []
    batch_indices: List[int] = []

    cache: Dict[int, Metrics] = {}

    logger.info("Metric computation started (%d frames)", total_frames)

    frame_idx = 0
    while True:
        ok, curr = cap.read()
        if not ok:
            break

        metric = Metrics(prev, curr, device=DEVICE, lpips_model=None)
        cache[frame_idx] = metric

        batch_t1.append(Metrics.lpips_preprocess(prev, DEVICE))
        batch_t2.append(Metrics.lpips_preprocess(curr, DEVICE))
        batch_indices.append(frame_idx)

        if len(batch_t1) == BATCH_SIZE:
            _flush_lpips_batch(batch_t1, batch_t2, batch_indices, cache, metrics)
            batch_t1.clear()
            batch_t2.clear()
            batch_indices.clear()

        prev = curr
        frame_idx += 1

        if progress_cb:
            progress_cb(
                frame_idx / (total_frames - 1),
                f"Computing metrics: {frame_idx}/{total_frames - 1}",
            )

    if batch_t1:
        _flush_lpips_batch(batch_t1, batch_t2, batch_indices, cache, metrics)

    cap.release()
    return metrics


def _flush_lpips_batch(
    t1: List[torch.Tensor],
    t2: List[torch.Tensor],
    indices: List[int],
    cache: Dict[int, Metrics],
    metrics: np.ndarray,
) -> None:
    """Run LPIPS on a batch and populate final metric array."""
    assert lpips_model is not None, "LPIPS model not initialized"

    with torch.no_grad():
        x1 = torch.cat(t1, dim=0).to(DEVICE)
        x2 = torch.cat(t2, dim=0).to(DEVICE)
        lpips_vals = lpips_model(x1, x2).squeeze().cpu().numpy()

    for j, idx in enumerate(indices):
        m = cache[idx]
        m.lpips = float(lpips_vals[j])
        m.difference = m.get_difference()

        metrics[idx] = (
            m.mse,
            m.inv_ssim,
            m.lpips,
            m.difference,
        )

# ---------------------------------------------------------------------
# HEALTHCHECK
# ---------------------------------------------------------------------
@app.get("/")
def health_check():
    return {
        "service": "Skip2Smooth Metrics API",
        "status": "running",
        "device": str(DEVICE),
    }

# ---------------------------------------------------------------------
# WEBSOCKET
# ---------------------------------------------------------------------
@app.websocket("/ws/compute-metrics")
async def ws_compute_metrics(ws: WebSocket):
    await ws.accept()

    async with GPU_SEMAPHORE:
        video_bytes = await ws.receive_bytes()
        video_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.mp4"
        video_path.write_bytes(video_bytes)

        queue: asyncio.Queue = asyncio.Queue()

        async def sender():
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                await ws.send_text(json.dumps(msg))

        sender_task = asyncio.create_task(sender())

        def progress_cb(progress: float, message: str):
            queue.put_nowait({"progress": progress, "message": message})

        try:
            metrics = await asyncio.to_thread(
                metrics_wrapper,
                video_path,
                progress_cb,
            )

            queue.put_nowait(None)
            await sender_task

            await ws.send_text(
                json.dumps(
                    {
                        "done": True,
                        "metrics": metrics.tolist(),
                    }
                )
            )

        finally:
            video_path.unlink(missing_ok=True)
            await ws.close()