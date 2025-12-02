"""
FastAPI server for Real-ESRGAN image upscaling
"""
import os
import cv2
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

app = FastAPI(title="Real-ESRGAN API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model caching
upsampler_cache = {}
face_enhancer_cache = None


class UpscaleRequest(BaseModel):
    model_name: str = "RealESRGAN_x4plus"
    scale: float = 4.0
    face_enhance: bool = False
    tile: int = 0
    denoise_strength: float = 0.5
    fp32: bool = False
    gpu_id: Optional[int] = None


def get_model_config(model_name: str):
    """Get model configuration based on model name"""
    model_name = model_name.split('.')[0]

    configs = {
        'RealESRGAN_x4plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'netscale': 4,
            'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        },
        'RealESRNet_x4plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            'netscale': 4,
            'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        },
        'RealESRGAN_x4plus_anime_6B': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
            'netscale': 4,
            'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        },
        'RealESRGAN_x2plus': {
            'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
            'netscale': 2,
            'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        },
        'realesr-animevideov3': {
            'model': SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
            'netscale': 4,
            'file_url': ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        },
        'realesr-general-x4v3': {
            'model': SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
            'netscale': 4,
            'file_url': [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]
        }
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(configs.keys())}")

    return configs[model_name]


def get_upsampler(model_name: str, tile: int = 0, fp32: bool = False, gpu_id: Optional[int] = None, denoise_strength: float = 0.5):
    """Get or create upsampler instance (with caching)"""
    cache_key = f"{model_name}_{tile}_{fp32}_{gpu_id}_{denoise_strength}"

    if cache_key in upsampler_cache:
        return upsampler_cache[cache_key]

    config = get_model_config(model_name)
    model = config['model']
    netscale = config['netscale']
    file_url = config['file_url']

    # Determine model path
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # Handle DNI (Denoise and Invert) for realesr-general-x4v3
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        if not os.path.isfile(wdn_model_path):
            wdn_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth'
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            wdn_model_path = load_file_from_url(
                url=wdn_url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # Create upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=not fp32,
        gpu_id=gpu_id
    )

    upsampler_cache[cache_key] = upsampler
    return upsampler


def get_face_enhancer(scale: float, upsampler):
    """Get or create face enhancer instance (with caching)"""
    global face_enhancer_cache

    if face_enhancer_cache is not None:
        return face_enhancer_cache

    from gfpgan import GFPGANer
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=scale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler
    )

    face_enhancer_cache = face_enhancer
    return face_enhancer


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Real-ESRGAN API Server",
        "version": "1.0.0",
        "endpoints": {
            "/": "This help message",
            "/health": "Health check",
            "/upscale": "POST endpoint for image upscaling (multipart/form-data)",
            "/models": "List available models",
            "/docs": "Swagger API documentation"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            "RealESRGAN_x4plus",
            "RealESRNet_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x2plus",
            "realesr-animevideov3",
            "realesr-general-x4v3"
        ],
        "default": "RealESRGAN_x4plus"
    }


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(..., description="Image file to upscale"),
    model_name: str = Form("RealESRGAN_x4plus", description="Model name"),
    scale: float = Form(4.0, description="Final upsampling scale"),
    face_enhance: bool = Form(False, description="Use GFPGAN to enhance face"),
    tile: int = Form(0, description="Tile size (0 for no tile)"),
    denoise_strength: float = Form(0.5, description="Denoise strength (only for realesr-general-x4v3)"),
    fp32: bool = Form(False, description="Use fp32 precision (default: fp16)"),
    gpu_id: Optional[int] = Form(None, description="GPU device ID")
):
    """
    Upscale an image using Real-ESRGAN

    - **file**: Image file (jpg, png, webp, etc.)
    - **model_name**: Model to use (see /models endpoint)
    - **scale**: Final upsampling scale (default: 4.0)
    - **face_enhance**: Enable face enhancement with GFPGAN (default: False)
    - **tile**: Tile size for large images (0 = auto-detect based on image size and GPU memory, default: 0)
    - **denoise_strength**: Denoise strength for realesr-general-x4v3 (0-1, default: 0.5)
    - **fp32**: Use full precision (slower but more accurate, default: False)
    - **gpu_id**: GPU device ID (None = auto, default: None)
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Determine image mode
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        # Auto-detect tile size if not specified and image is large
        # Estimate memory usage: for x4 upscale, output will be 16x larger in pixels
        # Rough estimate: if input > 500px on any side, use tiling
        h, w = img.shape[:2]
        max_dimension = max(h, w)
        auto_tile = tile

        if tile == 0 and max_dimension > 500:
            # Auto-enable tiling for large images
            # Tile size based on GPU memory: 512 for 12GB+, 400 for 8GB+, 200 for 4GB, 0 for CPU
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(gpu_id if gpu_id is not None else 0).total_memory / (1024**3)  # GB
                if gpu_memory >= 12:
                    auto_tile = 512  # RTX 3080 Ti, 3090, etc.
                elif gpu_memory >= 8:
                    auto_tile = 400
                elif gpu_memory >= 4:
                    auto_tile = 200
                else:
                    auto_tile = 0  # Will likely fail, but user can override
            else:
                auto_tile = 0  # CPU mode, no tiling needed

        # Get upsampler
        upsampler = get_upsampler(model_name, auto_tile, fp32, gpu_id, denoise_strength)

        # Process image
        try:
            if face_enhance:
                face_enhancer = get_face_enhancer(scale, upsampler)
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = upsampler.enhance(img, outscale=scale)
        except RuntimeError as error:
            raise HTTPException(
                status_code=500,
                detail=f"Processing error: {str(error)}. Try setting a smaller tile size."
            )

        # Save to temporary file
        extension = 'png' if img_mode == 'RGBA' else 'jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as tmp_file:
            cv2.imwrite(tmp_file.name, output)
            tmp_path = tmp_file.name

        return FileResponse(
            tmp_path,
            media_type=f'image/{extension}',
            filename=f'upscaled.{extension}',
            headers={"Content-Disposition": f'attachment; filename="upscaled.{extension}"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

