import os
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Ordered list of weight paths the engine will search automatically.
# Place your trained best.pt in the repo root as 'receipt_detector.pt'
# and it will be picked up first.
WEIGHT_SEARCH_PATHS = [
    "receipt_detector.pt",
    "runs/detect/train3/weights/best.pt",
    "runs/detect/train2/weights/best.pt",
    "runs/detect/train/weights/best.pt",
    "runs/detect/receipt_yolo26/weights/best.pt",
    "weights/best.pt",
]


class ReceiptEngine:
    def __init__(self, model_path: str = None, conf: float = 0.30, buffer: int = 20):
        """
        YOLO26 multi-receipt detection engine.

        Args:
            model_path: Explicit path to trained .pt weights.
                        If None, searches WEIGHT_SEARCH_PATHS automatically.
            conf:       Minimum detection confidence (0-1). Default 0.30.
            buffer:     Pixel padding added around each detected box. Default 20.
        """
        self.conf   = conf
        self.buffer = buffer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        resolved = self._resolve_path(model_path)
        if resolved is None:
            raise FileNotFoundError(
                "No trained YOLO26 weights found.\n"
                "Searched:\n" + "\n".join(f"  · {p}" for p in WEIGHT_SEARCH_PATHS) +
                "\n\nRename your trained best.pt to 'receipt_detector.pt' "
                "and place it in the repo root."
            )

        self.model_path = resolved
        self.model      = YOLO(resolved)
        print(f"[ReceiptEngine] Weights : {resolved}")
        print(f"[ReceiptEngine] Device  : {self.device}")
        print(f"[ReceiptEngine] Conf    : {conf}  Buffer: {buffer}px")

    # ── Internal helpers ───────────────────────────────────────

    @staticmethod
    def _resolve_path(model_path: str) -> str | None:
        candidates = ([model_path] if model_path else []) + WEIGHT_SEARCH_PATHS
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def _to_bgr(self, image) -> np.ndarray:
        """
        Normalise any image input into a BGR numpy array for YOLO/OpenCV.
        Accepts: file path (str) | PIL Image | numpy array (RGB or BGR).
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image: {image}")
            return img

        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                # Assume arrays coming from Streamlit/PIL are RGB
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image  # grayscale or already BGR

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _crop(self, img_bgr: np.ndarray, box) -> np.ndarray:
        """Apply buffer and crop one bounding box from img_bgr."""
        h, w   = img_bgr.shape[:2]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx1 = max(0,  int(x1) - self.buffer)
        cy1 = max(0,  int(y1) - self.buffer)
        cx2 = min(w,  int(x2) + self.buffer)
        cy2 = min(h,  int(y2) + self.buffer)
        return img_bgr[cy1:cy2, cx1:cx2]

    def _bgr_to_pil(self, img_bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # ── Public API ─────────────────────────────────────────────

    def detect_and_crop(self, image, return_pil: bool = True):
        """
        Detect the single highest-confidence receipt and return a crop.
        Preserves your original interface for backwards compatibility.

        Args:
            image:      File path, PIL Image, or numpy array.
            return_pil: True → PIL Image. False → BGR numpy array.

        Returns:
            Cropped image or None if nothing detected.
        """
        crops = self.detect_and_crop_all(image, return_pil=return_pil)
        return crops[0] if crops else None

    def detect_and_crop_all(self, image, return_pil: bool = True) -> list:
        """
        Detect ALL receipts in an image and return every crop.
        Sorted by confidence (highest first).

        Args:
            image:      File path, PIL Image, or numpy array.
            return_pil: True → list of PIL Images. False → list of BGR numpy arrays.

        Returns:
            List of cropped images (empty list if none detected).
        """
        img_bgr = self._to_bgr(image)

        results = self.model.predict(
            source  = img_bgr,
            conf    = self.conf,
            device  = self.device,
            verbose = False,
        )

        if not results or len(results[0].boxes) == 0:
            return []

        # Sort by confidence descending
        boxes = sorted(results[0].boxes, key=lambda b: float(b.conf[0]), reverse=True)

        crops = []
        for box in boxes:
            crop_bgr = self._crop(img_bgr, box)
            crops.append(self._bgr_to_pil(crop_bgr) if return_pil else crop_bgr)

        return crops

    def detect_boxes(self, image) -> list[dict]:
        """
        Return raw detection metadata without cropping.
        Useful for drawing overlays or debugging.

        Returns list of dicts:
            { 'x1', 'y1', 'x2', 'y2', 'conf', 'class_id' }
        """
        img_bgr = self._to_bgr(image)
        results = self.model.predict(
            source=img_bgr, conf=self.conf, device=self.device, verbose=False
        )
        if not results or len(results[0].boxes) == 0:
            return []

        out = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            out.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "conf": float(box.conf[0]),
                "class_id": int(box.cls[0]),
            })
        return out

    def annotate(self, image) -> Image.Image:
        """
        Return a copy of the image with detection boxes drawn on it.
        Useful for the Streamlit debug view.
        """
        img_bgr  = self._to_bgr(image).copy()
        boxes    = self.detect_boxes(img_bgr)
        h, w     = img_bgr.shape[:2]

        for i, b in enumerate(boxes):
            cv2.rectangle(img_bgr, (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                          (0, 229, 176), 2)
            label = f"receipt {i+1}  {b['conf']:.0%}"
            cv2.putText(img_bgr, label, (b["x1"], max(b["y1"] - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 176), 2)

        return self._bgr_to_pil(img_bgr)
