from pathlib import Path

import numpy as np
import onnxruntime as ort

DEFAULT_MODEL = Path(__file__).parent.parent / "napari_macrophage" / "models" / "best.onnx"


def load_session(model_path: Path = DEFAULT_MODEL) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _pct_norm(arr: np.ndarray, lo_pct: float = 0.1, hi_pct: float = 99.9) -> np.ndarray:
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def run_detection(
    session: ort.InferenceSession,
    cd206: np.ndarray,
    dapi: np.ndarray,
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """Pure inference: arrays in, list of per-slice detections out. No napari deps."""
    if cd206.ndim == 2:
        cd206 = cd206[np.newaxis]
    if dapi.ndim == 2:
        dapi = dapi[np.newaxis]

    Z, H, W = cd206.shape
    cd206_norm = _pct_norm(cd206)
    dapi_norm = _pct_norm(dapi)
    zeros = np.zeros((H, W), dtype=np.float32)

    input_name = session.get_inputs()[0].name
    results = []

    for z in range(Z):
        inp = np.stack([cd206_norm[z], zeros, dapi_norm[z]], axis=0)[np.newaxis]
        raw = session.run(None, {input_name: inp})[0]

        preds = np.squeeze(raw)
        if preds.ndim == 1:
            preds = preds[np.newaxis]
        if preds.ndim == 2 and preds.shape[0] == 5:
            preds = preds.T  # (5, N) → (N, 5)

        boxes = []
        for det in preds:
            if len(det) < 5:
                continue
            xc, yc, bw, bh, score = (
                float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
            )
            if score < confidence_threshold:
                continue
            x_min = max(0.0, xc - bw / 2)
            x_max = min(float(W - 1), xc + bw / 2)
            y_min = max(0.0, yc - bh / 2)
            y_max = min(float(H - 1), yc + bh / 2)
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append({"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "score": score})

        results.append({"z": z, "boxes": boxes})

    return results
