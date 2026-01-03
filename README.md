# Skip2Smooth GPU Metrics API

A high-performance GPU-accelerated service for computing video quality metrics, designed for evaluating frame interpolation results in the Skip2Smooth project.

## Overview

This FastAPI-based service provides real-time computation of multiple video quality metrics including MSE, SSIM, and LPIPS. It leverages NVIDIA GPU acceleration for efficient batch processing of video frames and offers WebSocket connectivity for streaming progress updates.

## Features

- **GPU-Accelerated Processing**: CUDA-enabled computation with NVIDIA GPU support
- **Multiple Quality Metrics**: Computes MSE, Inverse SSIM, and LPIPS in a single pass
- **Batched Computation**: Efficient batch processing for handling large videos
- **Real-time Progress**: WebSocket API with streaming progress updates
- **Containerized Deployment**: Docker support with NVIDIA CUDA runtime
- **Async Architecture**: Built on FastAPI for high-performance async operations

## Metrics Computed

The service computes four frame-wise metrics between consecutive video frames:

1. **MSE (Mean Squared Error)**: Pixel-level difference measurement
2. **Inverse SSIM**: 1 - Structural Similarity Index, measuring structural changes
3. **LPIPS**: Learned Perceptual Image Patch Similarity using AlexNet
4. **Weighted Difference**: Combined metric score (α×MSE + β×InvSSIM + γ×LPIPS)

Default weights: α=0.5, β=0.3, γ=0.2

## Requirements

### Hardware
- NVIDIA GPU with CUDA compute capability 3.5+
- Minimum 4GB GPU memory recommended
- Videos must contain at least 120 frames for GPU processing

### Software
- Docker with NVIDIA Container Toolkit
- CUDA 12.1+ compatible GPU drivers

## Installation

### Using Docker (Recommended)

```bash
# Pull the latest image
docker pull satvikvirmani/s2s-gpu-metrics:latest

# Run the container
docker run --gpus all -p 8000:8000 satvikvirmani/s2s-gpu-metrics:latest
```

### Using Docker Compose

```bash
docker compose up --build
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/s2s-gpu-metrics.git
cd s2s-gpu-metrics

# Build the Docker image
docker build -t s2s-gpu-metrics .

# Run the container
docker run --gpus all -p 8000:8000 s2s-gpu-metrics
```

## Usage

### Health Check

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "service": "Skip2Smooth Metrics API",
  "status": "running",
  "device": "cuda"
}
```

### WebSocket API

Connect to `ws://localhost:8000/ws/compute-metrics` and send video file bytes.

#### Example (Python)

```python
import asyncio
import websockets
import json

async def compute_metrics(video_path):
    uri = "ws://localhost:8000/ws/compute-metrics"
    
    async with websockets.connect(uri) as websocket:
        # Send video file
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        await websocket.send(video_bytes)
        
        # Receive progress updates and results
        async for message in websocket:
            data = json.loads(message)
            
            if 'progress' in data:
                print(f"Progress: {data['progress']*100:.1f}% - {data['message']}")
            
            if data.get('done'):
                metrics = data['metrics']
                print(f"Computation complete! Got {len(metrics)} frame metrics")
                return metrics

# Run
asyncio.run(compute_metrics("video.mp4"))
```

#### Response Format

Progress updates:
```json
{
  "progress": 0.45,
  "message": "Computing metrics: 540/1200"
}
```

Final result:
```json
{
  "done": true,
  "metrics": [
    [0.123, 0.045, 0.234, 0.156],
    [0.134, 0.048, 0.241, 0.163],
    ...
  ]
}
```

Each metrics array contains: `[MSE, Inverse SSIM, LPIPS, Weighted Difference]`

## Architecture

### Technology Stack

- **Base Image**: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
- **Framework**: FastAPI with Uvicorn ASGI server
- **Deep Learning**: PyTorch with CUDA 12.1 support
- **Computer Vision**: OpenCV, scikit-image
- **Perceptual Metrics**: LPIPS (AlexNet backbone)

### Processing Pipeline

1. Video file received via WebSocket
2. Frames extracted using OpenCV
3. Frame pairs processed in batches (default: 16 frames)
4. MSE and SSIM computed on CPU
5. LPIPS computed on GPU in batches
6. Results aggregated and returned as NumPy array

### GPU Optimization

- Single GPU semaphore to prevent memory overflow
- Batch processing for LPIPS computation
- Model loaded once at startup
- Efficient tensor operations with torch.no_grad()

## Configuration

Key constants in `main.py`:

```python
BATCH_SIZE = 16              # Frames per LPIPS batch
MIN_FRAMES_FOR_GPU = 120     # Minimum frames required
GPU_SEMAPHORE = 1            # Concurrent GPU operations
```

Metric weights in `metrics.py`:

```python
alpha = 0.5  # MSE weight
beta = 0.3   # Inverse SSIM weight
gamma = 0.2  # LPIPS weight
```

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally (requires CUDA-capable GPU)
uvicorn main:app --reload
```

### Running Tests

```bash
# Test with sample video
python test_client.py --video sample.mp4
```

## CI/CD

The repository includes GitHub Actions workflow for automated Docker builds:

- Triggered on push/PR to main branch
- Builds and pushes to Docker Hub
- Tags with commit SHA and 'latest'
- Utilizes build cache for faster iterations

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Container Toolkit is installed:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

Reduce batch size in `main.py`:
```python
BATCH_SIZE = 8  # or lower
```

### Video Too Small Error

The service requires at least 120 frames. For shorter videos, modify:
```python
MIN_FRAMES_FOR_GPU = 30  # or your desired minimum
```

## Contact

For issues and questions:
- GitHub Issues: [Repository Issues Page]
- Docker Hub: https://hub.docker.com/r/satvikvirmani/s2s-gpu-metrics
