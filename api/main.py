import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import tifffile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status

from .inference import DEFAULT_MODEL, load_session, run_detection
from .schemas import BoundingBox, DetectionResponse, HealthResponse, SliceDetections

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))
    if not model_path.exists():
        raise RuntimeError(f"ONNX model not found: {model_path}")
    _state["session"] = load_session(model_path)
    _state["model_name"] = model_path.name
    yield
    _state.clear()


app = FastAPI(
    title="Macrophage Detection API",
    description=(
        "RESTful API for ONNX-based macrophage detection in fluorescence microscopy images. "
        "Accepts CD206 and DAPI channel TIFFs and returns bounding-box detections per Z-slice."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Service health check",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded="session" in _state,
        model_name=_state.get("model_name", ""),
    )


@app.post(
    "/api/v1/detections",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    tags=["detection"],
    summary="Run macrophage detection on CD206 + DAPI TIFF images",
)
async def create_detections(
    cd206: UploadFile = File(..., description="CD206 channel TIFF (2D or 3D z-stack)"),
    dapi: UploadFile = File(..., description="DAPI channel TIFF (2D or 3D z-stack)"),
    confidence: float = Form(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for returned detections",
    ),
) -> DetectionResponse:
    try:
        cd206_arr = tifffile.imread(io.BytesIO(await cd206.read()))
        dapi_arr = tifffile.imread(io.BytesIO(await dapi.read()))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not read TIFF: {exc}",
        )

    if cd206_arr.shape != dapi_arr.shape:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Shape mismatch: cd206={cd206_arr.shape}, dapi={dapi_arr.shape}",
        )

    try:
        raw_results = run_detection(_state["session"], cd206_arr, dapi_arr, confidence)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    slices = [
        SliceDetections(z=r["z"], boxes=[BoundingBox(**b) for b in r["boxes"]])
        for r in raw_results
    ]
    total = sum(len(s.boxes) for s in slices)

    return DetectionResponse(
        model=_state["model_name"],
        confidence_threshold=confidence,
        total_boxes=total,
        slices=slices,
    )
