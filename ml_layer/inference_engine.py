from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE = 640
STRIDE = 32
LETTERBOX_COLOR = (114, 114, 114)

_INTRA_OP_THREADS = int(os.getenv("ORT_INTRA_OP_THREADS", "0")) or None
_INTER_OP_THREADS = int(os.getenv("ORT_INTER_OP_THREADS", "1"))


@dataclass
class Detection:
    xyxy: List[float]  # pixel coords in the ORIGINAL image
    confidence: float


class OnnxYoloEngine:

    def __init__(self, weights_path: str | Path, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        self.weights_path = str(weights_path)
        self.default_conf = conf_threshold
        self.iou_threshold = iou_threshold

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if _INTRA_OP_THREADS:
            so.intra_op_num_threads = _INTRA_OP_THREADS
        so.inter_op_num_threads = _INTER_OP_THREADS

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            self.weights_path, sess_options=so, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # -- preprocessing --------------------------------------------------
    @staticmethod
    def _letterbox(image_bgr: np.ndarray, new_shape: int = IMG_SIZE):
        """Resize+pad to a square, preserving aspect ratio (matches the
        preprocessing the model was trained/exported with)."""
        h, w = image_bgr.shape[:2]
        r = min(new_shape / h, new_shape / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
        dw /= 2
        dh /= 2

        if (w, h) != new_unpad:
            image_bgr = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            image_bgr, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=LETTERBOX_COLOR
        )
        return padded, r, (left, top)

    def _preprocess(self, image_rgb: np.ndarray):
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        padded, ratio, (pad_x, pad_y) = self._letterbox(image_bgr, IMG_SIZE)
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]  # NCHW
        return np.ascontiguousarray(img), ratio, (pad_x, pad_y)

    # -- postprocessing ---------------------------------------------------
    def _postprocess(self, output: np.ndarray, ratio: float, pad, orig_w, orig_h,
                      conf_threshold: float) -> List[Detection]:
        # output shape: (1, 4 + num_classes, num_anchors) -> (num_anchors, 4+num_classes)
        preds = output[0].transpose(1, 0)
        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:]
        class_scores = scores.max(axis=1)
        keep = class_scores >= conf_threshold
        if not np.any(keep):
            return []

        boxes_cxcywh = boxes_cxcywh[keep]
        class_scores = class_scores[keep]

        # cxcywh (model/letterbox space) -> xyxy (model space)
        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # undo letterbox padding + scale -> original image space
        pad_x, pad_y = pad
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        boxes_xywh_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        scores_list = class_scores.tolist()

        keep_idx = cv2.dnn.NMSBoxes(
            boxes_xywh_for_nms, scores_list,
            score_threshold=conf_threshold, nms_threshold=self.iou_threshold
        )
        if len(keep_idx) == 0:
            return []
        keep_idx = np.array(keep_idx).flatten()

        detections = []
        for i in keep_idx:
            detections.append(Detection(
                xyxy=[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                confidence=float(class_scores[i]),
            ))
        return detections

    # -- public API -------------------------------------------------------
    def predict(self, image_rgb: np.ndarray, conf_threshold: float | None = None) -> List[Detection]:
        conf = self.default_conf if conf_threshold is None else conf_threshold
        orig_h, orig_w = image_rgb.shape[:2]
        tensor, ratio, pad = self._preprocess(image_rgb)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return self._postprocess(output, ratio, pad, orig_w, orig_h, conf)

    def warmup(self):
        dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        self.predict(dummy, conf_threshold=0.99)