import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .georeference import bbox_to_geojson_polygon, CornerValidationError
from .export_payload import build_backend_payload
from .inference_engine import OnnxYoloEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oil_spill_api")

CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = CURRENT_DIR / "weights" / "best.onnx"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
TIMESTAMP_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S")

app = FastAPI(title="Oil Spill Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

engine: OnnxYoloEngine | None = None


@app.on_event("startup")
def _load_and_warm_model() -> None:
    global engine
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")
    engine = OnnxYoloEngine(WEIGHTS_PATH, conf_threshold=0.5)
    engine.warmup()
    logger.info("Model loaded and warmed up: %s", WEIGHTS_PATH)


def _parse_timestamp(capture_time: str) -> datetime:
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(capture_time, fmt)
        except ValueError:
            continue
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Invalid capture_time format '{capture_time}'. Supported: YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SSZ, YYYY-MM-DD HH:MM:SS",
    )


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_spill(
    file: UploadFile = File(..., description="Satellite image (.jpg or .png)"),
    capture_time: str = Form(..., description="Satellite capture timestamp"),
    ul_lat: float = Form(...),
    ul_lon: float = Form(...),
    ur_lat: float = Form(...),
    ur_lon: float = Form(...),
    bl_lat: float = Form(...),
    bl_lon: float = Form(...),
    br_lat: float = Form(...),
    br_lon: float = Form(...),
    conf_threshold: float = Form(0.50, ge=0.0, le=1.0),
) -> Dict[str, Any]:

    if engine is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is still loading.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported content type '{file.content_type}'.",
        )

    parsed_time = _parse_timestamp(capture_time)

    corners = {
        "ul": (ul_lon, ul_lat),
        "ur": (ur_lon, ur_lat),
        "bl": (bl_lon, bl_lat),
        "br": (br_lon, br_lat),
    }

    try:
        _validate_corners(corners)
    except CornerValidationError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))

    raw = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "File exceeds the 20 MB limit.",
        )

    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
        pil_img = pil_img.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "File is not a valid image.",
        )

    image_rgb = np.asarray(pil_img)
    img_height, img_width = image_rgb.shape[:2]

    try:
        detections = engine.predict(image_rgb, conf_threshold=conf_threshold)
    except Exception:
        logger.exception("Inference failed for file=%s", file.filename)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Inference failed.",
        )

    payloads: List[Dict[str, Any]] = []

    for det in detections:
        geojson_poly = bbox_to_geojson_polygon(
            det.xyxy,
            img_width,
            img_height,
            corners,
        )

        wgs84_coords = geojson_poly["coordinates"][0]

        payloads.append(
            build_backend_payload(
                wgs84_coords,
                parsed_time,
                det.confidence,
                file.filename,
            )
        )

    return {
        "status": "success",
        "filename": file.filename,
        "detected_at": parsed_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spills_found": len(payloads),
        "detections": payloads,
    }


def _validate_corners(corners: Dict[str, tuple]) -> None:
    lats = [v[1] for v in corners.values()]
    lons = [v[0] for v in corners.values()]

    if any(not (-90.0 <= lat <= 90.0) for lat in lats):
        raise CornerValidationError("Corner latitude out of range [-90, 90].")

    if any(not (-180.0 <= lon <= 180.0) for lon in lons):
        raise CornerValidationError("Corner longitude out of range [-180, 180].")

    if max(lons) - min(lons) > 180.0:
        raise CornerValidationError(
            "Corner longitudes span more than 180 degrees."
        )


@app.get("/health")
async def health_check():
    return {"status": "active", "model_loaded": engine is not None}


@app.get("/")
async def root():
    return {"message": "Oil Spill Detection API is running. Send POST requests to /predict."}