from __future__ import annotations
from typing import List, Tuple, cast

import cv2, numpy as np, re
import easyocr
from schema import GraphMessage
from langchain_core.runnables import RunnableConfig

# 1️⃣  EasyOCR reader – detector ON for better localisation
reader = easyocr.Reader(
    ['en'], gpu=False, detector=True, recog_network='standard'
)

# ─── helper: crop candidate plate region ──────────────────────────
def _crop_plate(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Heuristically crop the largest 4‑corner contour that looks like a
    licence plate. Returns (crop, found_flag).
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    best, best_area = None, 0

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect = w / h if h else 0
        if 2 < aspect < 6 and 0.02 * img_area < area < 0.5 * img_area:
            if area > best_area:
                best_area = area
                best = (x, y, w, h)

    if best:
        x, y, w, h = best
        return img[y:y + h, x:x + w], True
    return img, False  # fallback to full image


# ─── agent function ───────────────────────────────────────────────
def ocr_agent(
    state: GraphMessage,
    _config: RunnableConfig | None = None
) -> GraphMessage:
    msg = state

    if msg.data is None or "image_path" not in msg.data:
        return GraphMessage(
            role="assistant",
            text="⚠️ No image received. Please upload a licence‑plate photo."
        )

    img_path: str = cast(str, msg.data["image_path"])
    img = cv2.imread(img_path)

    # 🔒  Guard: imread failed
    if img is None:
        return GraphMessage(
            role="assistant",
            text="⚠️ Failed to read that image file. Please try another photo."
        )

    # 2️⃣  Crop likely plate region
    crop, found = _crop_plate(img)

    # 3️⃣  Run EasyOCR
    def _run_ocr(mat: np.ndarray) -> List[str]:
        return [r for r in reader.readtext(mat, detail=0) if isinstance(r, str)]

    texts = _run_ocr(crop)
    if not texts and found:
        texts = _run_ocr(img)  # fallback to full frame

    plate_candidate = max(texts, key=len) if texts else ""
    plate_clean = re.sub(r"[^A-Z0-9]", "", plate_candidate.upper())

    if not plate_clean:
        return GraphMessage(
            role="assistant",
            text="Sorry, I couldn’t read a licence‑plate from that image."
        )

    # 4️⃣  Delegate to ProcessingAgent
    return GraphMessage(
        role="delegate",
        text="process",
        data={"plate": plate_clean}
    )
