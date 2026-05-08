from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class ImagePreprocessResult:
    image: Image.Image
    blur_score: float
    quality_score: float
    was_cropped: bool = False
    was_deskewed: bool = False
    deskew_angle: float = 0.0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _cv_to_pil(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:
        return Image.fromarray(image).convert("RGB")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGB")


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def _four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = _order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(height_a), int(height_b))

    if max_width < 100 or max_height < 100:
        return image

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def detect_blur_score(image: Image.Image) -> float:
    """
    Higher score means sharper image.
    Lower score means blurry image.
    """
    cv_image = _pil_to_cv(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(score)


def crop_document_boundary(image: Image.Image) -> tuple[Image.Image, bool, list[str]]:
    """
    Detect the largest document-like rectangle and crop it.
    If no clear document boundary is found, returns the original image.
    """
    warnings: list[str] = []

    cv_image = _pil_to_cv(image)
    original = cv_image.copy()
    height, width = cv_image.shape[:2]

    if width < 300 or height < 300:
        warnings.append("Image resolution is low; document boundary detection may be inaccurate.")
        return image, False, warnings

    ratio = height / 700.0
    resized = cv2.resize(cv_image, (int(width / ratio), 700))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    document_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            document_contour = approx.reshape(4, 2)
            break

    if document_contour is None:
        warnings.append("No clear document boundary detected; using original image.")
        return image, False, warnings

    scaled_points = document_contour.astype("float32") * ratio
    warped = _four_point_transform(original, scaled_points)

    return _cv_to_pil(warped), True, warnings


def deskew_image(image: Image.Image) -> tuple[Image.Image, bool, float, list[str]]:
    """
    Deskew a tilted document image.
    """
    warnings: list[str] = []

    cv_image = _pil_to_cv(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )[1]

    coordinates = np.column_stack(np.where(thresh > 0))

    if len(coordinates) < 50:
        warnings.append("Not enough text pixels detected for deskewing.")
        return image, False, 0.0, warnings

    angle = cv2.minAreaRect(coordinates)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image, False, float(angle), warnings

    height, width = cv_image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        cv_image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return _cv_to_pil(rotated), True, float(angle), warnings


def enhance_for_ocr(image: Image.Image) -> Image.Image:
    """
    Improve contrast and reduce noise before AI OCR.
    """
    cv_image = _pil_to_cv(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return _cv_to_pil(enhanced)


def calculate_quality_score(blur_score: float, was_cropped: bool, was_deskewed: bool) -> float:
    """
    Calculate image quality score from 0 to 1.
    This is not the OCR model confidence. It is image quality confidence.
    """
    blur_quality = min(1.0, blur_score / 250.0)

    score = blur_quality

    if was_cropped:
        score += 0.05

    if was_deskewed:
        score += 0.05

    return round(max(0.0, min(1.0, score)), 3)


def preprocess_for_ocr(
    image: Image.Image,
    page_number: int,
    blur_threshold: float = 80.0,
) -> ImagePreprocessResult:
    warnings: list[str] = []

    original_blur_score = detect_blur_score(image)

    if original_blur_score < blur_threshold:
        warnings.append(
            f"Image appears blurry. Blur score={original_blur_score:.2f}, threshold={blur_threshold:.2f}."
        )

    cropped_image, was_cropped, crop_warnings = crop_document_boundary(image)
    warnings.extend(crop_warnings)

    deskewed_image, was_deskewed, deskew_angle, deskew_warnings = deskew_image(cropped_image)
    warnings.extend(deskew_warnings)

    enhanced_image = enhance_for_ocr(deskewed_image)

    final_blur_score = detect_blur_score(enhanced_image)

    quality_score = calculate_quality_score(
        blur_score=final_blur_score,
        was_cropped=was_cropped,
        was_deskewed=was_deskewed,
    )

    if quality_score < 0.5:
        warnings.append(
            f"Low image quality detected after preprocessing. Quality score={quality_score:.2f}."
        )

    return ImagePreprocessResult(
        image=enhanced_image,
        blur_score=round(final_blur_score, 3),
        quality_score=quality_score,
        was_cropped=was_cropped,
        was_deskewed=was_deskewed,
        deskew_angle=round(deskew_angle, 3),
        warnings=warnings,
        metadata={
            "page_number": page_number,
            "original_blur_score": round(original_blur_score, 3),
            "final_blur_score": round(final_blur_score, 3),
            "quality_score": quality_score,
            "was_cropped": was_cropped,
            "was_deskewed": was_deskewed,
            "deskew_angle": round(deskew_angle, 3),
        },
    )