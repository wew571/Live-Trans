"""Process entry point for the live OCR and translation models."""

from __future__ import annotations

from math import ceil, floor
import os
from pathlib import Path
import sys

from live_models import _is_translation_batch, read_packet, write_packet
from live_translator import load_translator
from core.setup_models import resolve_model

_ROOT = Path(__file__).resolve().parents[1]


def _load_ocr():
    try:
        import cv2
        import torch
        from paddleocr import PaddleOCR
    except ImportError as error:
        raise RuntimeError(f"Install the live PaddleOCR 3.7 and Transformers 5.23 runtime: {error}") from error

    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    detection = "PP-OCRv6_small_det"
    recognition = "PP-OCRv6_small_rec"
    use_cuda = torch.cuda.is_available()
    ocr = PaddleOCR(
        device="gpu:0" if use_cuda else "cpu",
        text_detection_model_name=detection,
        text_detection_model_dir=resolve_model(_ROOT, "live-ocr-det"),
        text_recognition_model_name=recognition,
        text_recognition_model_dir=resolve_model(_ROOT, "live-ocr-rec"),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_batch_size=8,
        cpu_threads=4,
        enable_mkldnn=False,
        engine="transformers",
        engine_config={"dtype": "float16" if use_cuda else "float32"},
    )
    if use_cuda:
        processor = ocr.paddlex_pipeline._pipeline.text_det_model.image_processor
        postprocess = processor.post_process_object_detection

        def postprocess_fp32(predictions, *args, **kwargs):
            # The OpenCV contour scoring routine requires a float32 probability map.
            predictions.last_hidden_state = predictions.last_hidden_state.float()
            return postprocess(predictions, *args, **kwargs)

        processor.post_process_object_detection = postprocess_fp32

    return ocr


def _decode_frame(request, pixels):
    import numpy as np

    width, height = request.get("width"), request.get("height")
    if (
        type(width) is not int
        or type(height) is not int
        or width < 1
        or height < 1
        or len(pixels) != height * width * 3
    ):
        raise ValueError("Invalid live OCR frame dimensions")
    return np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))


def _recognize(ocr, frame):
    height, width = frame.shape[:2]
    rows = []
    results = ocr.predict(frame, text_det_limit_side_len=960, text_det_limit_type="max")
    for result in results:
        for polygon, value in zip(result["rec_polys"], result["rec_texts"], strict=True):
            text = value.strip()
            if not text:
                continue
            x, y = zip(*polygon)
            left = max(0, floor(min(x)))
            top = max(0, floor(min(y)))
            right = min(width, ceil(max(x)))
            bottom = min(height, ceil(max(y)))
            if left < right and top < bottom:
                rows.append(((left, top, right, bottom), text))
    return rows


def serve(input_stream, output_stream):
    ocr = None
    translator = None
    while (packet := read_packet(input_stream)) is not None:
        request, binary = packet
        try:
            operation = request.get("op")
            if operation == "ocr":
                frame = _decode_frame(request, binary)
                if ocr is None:
                    ocr = _load_ocr()
                result = _recognize(ocr, frame)
            elif operation == "translate":
                texts = request.get("texts")
                if binary or not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
                    raise ValueError("Live translation expects a list of strings")
                if translator is None:
                    translator = load_translator()
                result = translator(texts)
                if not _is_translation_batch(result, len(texts)):
                    raise ValueError("The live translator must return one string per input")
            else:
                raise ValueError("Unknown live inference request")
            write_packet(output_stream, {"ok": True, "result": result})
        except BrokenPipeError:
            return
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"[:2000]
            write_packet(output_stream, {"ok": False, "error": detail})
            return


def main():
    # Libraries may log with print() or through native code. Reserve a duplicate
    # of the stdout pipe for the protocol and route descriptor 1 to stderr.
    output_fd = os.dup(sys.stdout.fileno())
    with os.fdopen(output_fd, "wb") as output_stream:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        serve(sys.stdin.buffer, output_stream)


if __name__ == "__main__":
    main()
