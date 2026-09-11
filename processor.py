from pathlib import Path
import cv2
import numpy as np


# =========================================================
# REMOVE RULED LINES
# =========================================================

def remove_ruled_lines(img: np.ndarray) -> np.ndarray:
    """Detects and erases horizontal notebook lines while preserving text."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert grayscale image
    inverted = cv2.bitwise_not(gray)

    # Detect horizontal lines
    kernel_width = max(15, img.shape[1] // 40)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1)
    )

    detected_lines = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=2
    )

    # Dilate detected lines slightly
    line_dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 3)
    )

    detected_lines = cv2.dilate(
        detected_lines,
        line_dilation_kernel,
        iterations=1
    )

    # Remove lines
    lines_removed = cv2.inpaint(
        gray,
        detected_lines,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    # Clean white background + black text
    cleaned = cv2.adaptiveThreshold(
        lines_removed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10,
    )

    return cleaned


# =========================================================
# CREATE LETTERHEAD / BANNER
# =========================================================

def add_letterhead(content: np.ndarray) -> np.ndarray:
    """
    Adds a clearly visible professional letterhead above the processed image.
    """

    # -----------------------------------------------------
    # Convert grayscale content to BGR
    # -----------------------------------------------------

    if len(content.shape) == 2:
        content = cv2.cvtColor(
            content,
            cv2.COLOR_GRAY2BGR
        )

    content_height, content_width = content.shape[:2]

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    REFERENCE_WIDTH = 1600

    scale = content_width / REFERENCE_WIDTH

    scale = max(0.4, min(scale, 3.0))

    def s(value):
        return int(round(value * scale))

    # -----------------------------------------------------
    # Letterhead dimensions
    # -----------------------------------------------------

    # Increased from 260 to 330
    header_height = s(380)

    canvas_width = content_width

    total_height = header_height + content_height

    canvas = np.full(
        (total_height, canvas_width, 3),
        255,
        dtype=np.uint8
    )

    # -----------------------------------------------------
    # HEADER CONTENT
    # -----------------------------------------------------

    company_name = "JAY DEEPTHI INTERIORS"
    subtitle = "INTERIOR DESIGN & DECORATION"
    address = "Nallapadu, Near HP Gas Booking Office, Guntur - 522005"
    phone = "Phone: +91 99518 23287"

    font_bold = cv2.FONT_HERSHEY_SIMPLEX
    font_regular = cv2.FONT_HERSHEY_SIMPLEX

    # =====================================================
    # COMPANY NAME
    # =====================================================

    # Instead of using a fixed font size,
    # calculate the font size based on page width.

    target_width = int(content_width * 0.88)

    company_scale = 1.0 * scale
    company_thickness = max(2, s(4))

    # Find the largest font that fits within 88% of page width
    while True:

        (text_width, text_height), _ = cv2.getTextSize(
            company_name,
            font_bold,
            company_scale,
            company_thickness
        )

        if text_width >= target_width:
            company_scale -= 0.05 * scale
            break

        company_scale += 0.05 * scale

        if company_scale > 5:
            break

    (text_width, text_height), _ = cv2.getTextSize(
        company_name,
        font_bold,
        company_scale,
        company_thickness
    )

    company_x = (canvas_width - text_width) // 2

    # Much larger heading position
    company_y = s(100)

    cv2.putText(
        canvas,
        company_name,
        (company_x, company_y),
        font_bold,
        company_scale,
        (25, 25, 25),
        company_thickness,
        cv2.LINE_AA
    )

    # =====================================================
    # SUBTITLE
    # =====================================================

    subtitle_target_width = int(canvas_width * 0.78)

    subtitle_scale = 1.05 * scale
    subtitle_thickness = max(2, s(3))

    while True:
        (text_width, text_height), _ = cv2.getTextSize(
        subtitle,
        font_regular,
        subtitle_scale,
        subtitle_thickness
    )
        if text_width >= subtitle_target_width:
            break

        subtitle_scale += 0.05 * scale

        if subtitle_scale > 5:
            break

    (text_width, text_height), _ = cv2.getTextSize(
    subtitle,
    font_regular,
    subtitle_scale,
    subtitle_thickness
)
    subtitle_x = (canvas_width - text_width) // 2
    subtitle_y = s(165)

    cv2.putText(
    canvas,
    subtitle,
    (subtitle_x, subtitle_y),
    font_regular,
    subtitle_scale,
    (60, 60, 60),
    subtitle_thickness,
    cv2.LINE_AA
)

    # =====================================================
    # ADDRESS
    # =====================================================

    address_target_width = int(canvas_width * 0.78)

    address_scale = 1.0 * scale
    address_thickness = max(3, s(3))

    while True:

        (text_width, text_height), _ = cv2.getTextSize(
        address,
        font_regular,
        address_scale,
        address_thickness
        )

        if text_width >= address_target_width:
            break

        address_scale += 0.05 * scale

        if address_scale > 5:
         break

        # Recalculate final size
    (text_width, text_height), _ = cv2.getTextSize(
    address,
    font_regular,
    address_scale,
    address_thickness
    )

    address_x = (canvas_width - text_width) // 2
    address_y = s(225)

    cv2.putText(
    canvas,
    address,
    (address_x, address_y),
    font_regular,
    address_scale,
    (60, 60, 60),
    address_thickness,
    cv2.LINE_AA
    )

    # =====================================================
    # PHONE
    # =====================================================

    phone_target_width = int(canvas_width * 0.45)

    phone_scale = 1.0 * scale
    phone_thickness = max(3, s(3))

    while True:

        (text_width, text_height), _ = cv2.getTextSize(
        phone,
        font_regular,
        phone_scale,
        phone_thickness
        )

        if text_width >= phone_target_width:
            break

        phone_scale += 0.05 * scale

        if phone_scale > 5:
            break

    # Recalculate final size
    (text_width, text_height), _ = cv2.getTextSize(
    phone,
    font_regular,
    phone_scale,
    phone_thickness
    )

    phone_x = (canvas_width - text_width) // 2
    phone_y = s(285)

    cv2.putText(
    canvas,
    phone,
    (phone_x, phone_y),
    font_regular,
    phone_scale,
    (60, 60, 60),
    phone_thickness,
    cv2.LINE_AA
    )

    # =====================================================
    # HORIZONTAL SEPARATOR
    # =====================================================

    line_y = s(325)

    margin = s(30)

    cv2.line(
        canvas,
        (margin, line_y),
        (canvas_width - margin, line_y),
        (40, 40, 40),
        max(2, s(3))
    )

    # =====================================================
    # PUT PROCESSED CONTENT UNDER HEADER
    # =====================================================

    canvas[
        header_height:
        header_height + content_height,
        0:content_width
    ] = content

    return canvas