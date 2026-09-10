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
    Adds a professional letterhead/banner above the processed image.
    Scales proportionally to content_width so it looks correct whether
    the source image is a desktop-resolution scan or a mobile photo.
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
    # Scale factor: everything below is tuned for a
    # reference width of 1600px, then scaled to whatever
    # width the actual image is (mobile photos are often
    # much wider/narrower than a desktop scan).
    # -----------------------------------------------------
 
    REFERENCE_WIDTH = 1600
    scale = content_width / REFERENCE_WIDTH
 
    # Clamp so extremely tiny or huge images don't produce
    # unreadable or absurdly large text
    scale = max(0.4, min(scale, 3.0))
 
    def s(value):
        """Scale a pixel value."""
        return int(round(value * scale))
 
    # -----------------------------------------------------
    # Letterhead dimensions
    # -----------------------------------------------------
 
    header_height = s(260)
 
    # Keep same width as original content
    canvas_width = content_width
 
    # Total output height
    total_height = header_height + content_height
 
    # Create pure white canvas
    canvas = np.full(
        (total_height, canvas_width, 3),
        255,
        dtype=np.uint8
    )
 
    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------
 
    company_name = "JAY DEEPTHI INTERIORS"
    subtitle = "INTERIOR DESIGN & DECORATION"
    address = "12-45, Main Road, Vijayawada, Andhra Pradesh"
    phone = "Phone: +91 98765 43210"
 
    font_bold = cv2.FONT_HERSHEY_SIMPLEX
    font_regular = cv2.FONT_HERSHEY_SIMPLEX
 
    # -----------------------------------------------------
    # Company name
    # -----------------------------------------------------
 
    company_scale = 1.5 * scale
    company_thickness = max(1, s(3))
 
    (text_width, text_height), _ = cv2.getTextSize(
        company_name,
        font_bold,
        company_scale,
        company_thickness
    )
 
    company_x = (canvas_width - text_width) // 2
    company_y = s(65)
 
    cv2.putText(
        canvas,
        company_name,
        (company_x, company_y),
        font_bold,
        company_scale,
        (30, 30, 30),
        company_thickness,
        cv2.LINE_AA
    )
 
    # -----------------------------------------------------
    # Subtitle
    # -----------------------------------------------------
 
    subtitle_scale = 0.75 * scale
    subtitle_thickness = max(1, s(2))
 
    (text_width, text_height), _ = cv2.getTextSize(
        subtitle,
        font_regular,
        subtitle_scale,
        subtitle_thickness
    )
 
    subtitle_x = (canvas_width - text_width) // 2
    subtitle_y = s(105)
 
    cv2.putText(
        canvas,
        subtitle,
        (subtitle_x, subtitle_y),
        font_regular,
        subtitle_scale,
        (80, 80, 80),
        subtitle_thickness,
        cv2.LINE_AA
    )
 
    # -----------------------------------------------------
    # Address
    # -----------------------------------------------------
 
    address_scale = 0.55 * scale
    address_thickness = max(1, s(1))
 
    (text_width, text_height), _ = cv2.getTextSize(
        address,
        font_regular,
        address_scale,
        address_thickness
    )
 
    address_x = (canvas_width - text_width) // 2
    address_y = s(150)
 
    cv2.putText(
        canvas,
        address,
        (address_x, address_y),
        font_regular,
        address_scale,
        (80, 80, 80),
        address_thickness,
        cv2.LINE_AA
    )
 
    # -----------------------------------------------------
    # Phone
    # -----------------------------------------------------
 
    phone_scale = 0.55 * scale
    phone_thickness = max(1, s(1))
 
    (text_width, text_height), _ = cv2.getTextSize(
        phone,
        font_regular,
        phone_scale,
        phone_thickness
    )
 
    phone_x = (canvas_width - text_width) // 2
    phone_y = s(185)
 
    cv2.putText(
        canvas,
        phone,
        (phone_x, phone_y),
        font_regular,
        phone_scale,
        (80, 80, 80),
        phone_thickness,
        cv2.LINE_AA
    )
 
    # -----------------------------------------------------
    # Horizontal separator
    # -----------------------------------------------------
 
    line_y = s(220)
    margin = s(30)
 
    cv2.line(
        canvas,
        (margin, line_y),
        (canvas_width - margin, line_y),
        (50, 50, 50),
        max(1, s(2))
    )
 
    # -----------------------------------------------------
    # Put processed content underneath header
    # -----------------------------------------------------
 
    canvas[
        header_height:
        header_height + content_height,
        0:content_width
    ] = content
 
    return canvas
