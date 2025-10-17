"""
Roboflow Hosted Inference API Detector

Provides a detector compatible with the BoxDetector interface but powered by
Roboflow's hosted detection API.
"""

from __future__ import annotations

import os
import io
import cv2
import json
import time
import base64
import logging
import requests
import numpy as np
from typing import Dict, List, Optional


class RoboflowAPIDetector:
    """
    Detector using Roboflow Hosted Inference API.

    Returns the same shape as BoxDetector.detect_boxes to keep the app logic consistent:
    {
        'boxes': List[[x1, y1, x2, y2]],
        'confidences': List[float],
        'count': int,
        'crops': Optional[List[np.ndarray]],
        'image_shape': Tuple[int, int, int]
    }
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        confidence: float = 0.5,
        overlap: float = 0.45,
        session: Optional[requests.Session] = None,
        timeout: int = 30,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.confidence = float(confidence)
        self.overlap = float(overlap)
        self.timeout = timeout
        self.session = session or requests.Session()

        self.logger = logging.getLogger(__name__)

    def _encode_image_to_jpeg(self, image: np.ndarray) -> bytes:
        """Encode an RGB or BGR numpy image to JPEG bytes."""
        if image is None:
            raise ValueError("Image is None")

        # Ensure BGR for OpenCV encoding
        if image.ndim == 3 and image.shape[2] == 3:
            # If image likely RGB (heuristic), convert to BGR just to be safe
            # We'll assume image is already BGR if coming from cv2; from Streamlit we pass RGB.
            # Try convert RGB->BGR by swapping channels (cheap and safe)
            image_bgr = image[:, :, ::-1]
        else:
            image_bgr = image

        success, buffer = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            raise RuntimeError("Failed to encode image to JPEG")
        return buffer.tobytes()

    def _post_image(self, image_bytes: bytes) -> Dict:
        """POST image bytes to Roboflow API and return parsed JSON."""
        params = {
            "api_key": self.api_key,
            "confidence": max(0.0, min(1.0, self.confidence)),
            "overlap": max(0.0, min(1.0, self.overlap)),
            "format": "json",
        }

        files = {
            "file": ("image.jpg", image_bytes, "image/jpeg"),
        }

        resp = self.session.post(self.api_url, params=params, files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def detect_boxes(self, image: np.ndarray, return_crops: bool = False) -> Dict:
        """
        Detect boxes via Roboflow API.
        """
        if not self.api_key or self.api_key.strip().upper() in {"", "YOUR_API_KEY", "YOUR_API_KEY_HERE", "<API_KEY>", "PLACEHOLDER"}:
            raise ValueError("Roboflow API key missing. Set ROBOFLOW_API_KEY or enter it in the sidebar.")

        try:
            img_bytes = self._encode_image_to_jpeg(image)
            data = self._post_image(img_bytes)

            boxes: List[List[int]] = []
            confidences: List[float] = []
            crops: List[np.ndarray] = []

            preds = data.get("predictions", [])

            img_h = int(data.get("image", {}).get("height", image.shape[0]))
            img_w = int(data.get("image", {}).get("width", image.shape[1]))

            for p in preds:
                # Roboflow returns center-based boxes
                x_c = float(p.get("x", 0.0))
                y_c = float(p.get("y", 0.0))
                w = float(p.get("width", 0.0))
                h = float(p.get("height", 0.0))
                conf = float(p.get("confidence", 0.0))

                x1 = int(round(x_c - w / 2))
                y1 = int(round(y_c - h / 2))
                x2 = int(round(x_c + w / 2))
                y2 = int(round(y_c + h / 2))

                # Clip to image bounds
                x1 = max(0, min(img_w - 1, x1))
                y1 = max(0, min(img_h - 1, y1))
                x2 = max(0, min(img_w - 1, x2))
                y2 = max(0, min(img_h - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)

                if return_crops:
                    crop = image[y1:y2, x1:x2]
                    crops.append(crop)

            return {
                "boxes": boxes,
                "confidences": confidences,
                "count": len(boxes),
                "crops": crops if return_crops else None,
                "image_shape": (img_h, img_w, image.shape[2] if image.ndim == 3 else 1),
            }
        except requests.HTTPError as http_err:
            # Try to extract error detail
            detail = None
            try:
                detail = http_err.response.json()
            except Exception:
                detail = http_err.response.text if http_err.response is not None else str(http_err)
            self.logger.error(f"Roboflow API HTTP error: {detail}")
            raise
        except Exception as e:
            self.logger.error(f"Roboflow API detection error: {e}")
            raise

    def visualize_detections(self, image: np.ndarray, detection_results: Dict) -> np.ndarray:
        """Draw bounding boxes on the image similar to BoxDetector.visualize_detections."""
        vis_image = image.copy()
        boxes = detection_results.get("boxes", [])
        confidences = detection_results.get("confidences", [])

        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"Box {i+1}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 200, 255), -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        count_text = f"Total Boxes: {len(boxes)}"
        cv2.putText(vis_image, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        return vis_image
