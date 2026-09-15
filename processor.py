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
    Premium interior-design letterhead.

    Cosmetic changes:
        - Deep wine/maroon brand color
        - Warm ivory background
        - Champagne-gold accents
        - Soft tinted header
        - Premium separator
        - Subtle geometric decoration
    """

    # =====================================================
    # CONVERT GRAYSCALE CONTENT TO BGR
    # =====================================================

    if len(content.shape) == 2:
        content = cv2.cvtColor(
            content,
            cv2.COLOR_GRAY2BGR
        )

    content_height, content_width = content.shape[:2]

    # =====================================================
    # SCALING
    # =====================================================

    REFERENCE_WIDTH = 1600

    scale = content_width / REFERENCE_WIDTH
    scale = max(0.4, min(scale, 3.0))

    def s(value):
        return max(1, int(round(value * scale)))

    # =====================================================
    # PREMIUM COLOR PALETTE
    # =====================================================

    # OpenCV = BGR

    # Deep premium wine / maroon
    MAROON = (72, 12, 105)

    # Darker wine
    DARK_MAROON = (48, 5, 75)

    # Warm ivory
    IVORY = (248, 244, 235)

    # Slightly darker ivory
    WARM_IVORY = (238, 231, 215)

    # Champagne gold
    GOLD = (170, 150, 95)

    # Light champagne
    LIGHT_GOLD = (210, 195, 155)

    # Dark warm gray
    CHARCOAL = (55, 52, 48)

    # Soft gray
    SOFT_GRAY = (115, 108, 98)

    WHITE = (255, 255, 255)

    # =====================================================
    # LETTERHEAD DIMENSIONS
    # =====================================================

    header_height = s(420)

    canvas_width = content_width

    total_height = header_height + content_height

    # =====================================================
    # PAGE BACKGROUND
    # =====================================================

    canvas = np.full(
        (total_height, canvas_width, 3),
        WHITE,
        dtype=np.uint8
    )

    # =====================================================
    # HEADER BACKGROUND
    # =====================================================

    # Warm ivory header
    canvas[
        0:header_height,
        0:canvas_width
    ] = IVORY

    # =====================================================
    # SUBTLE DECORATIVE MAROON SHAPE
    # =====================================================

    # Right-side soft geometric accent
    cv2.ellipse(
        canvas,
        (
            canvas_width + s(80),
            s(60)
        ),
        (s(260), s(260)),
        0,
        0,
        360,
        WARM_IVORY,
        -1
    )

    cv2.ellipse(
        canvas,
        (
            canvas_width + s(20),
            s(80)
        ),
        (s(170), s(170)),
        0,
        0,
        360,
        LIGHT_GOLD,
        2
    )

    # Left-side subtle accent
    cv2.ellipse(
        canvas,
        (
            -s(100),
            s(330)
        ),
        (s(180), s(180)),
        0,
        0,
        360,
        WARM_IVORY,
        -1
    )

    # =====================================================
    # FONTS
    # =====================================================

    font_bold = cv2.FONT_HERSHEY_SIMPLEX
    font_regular = cv2.FONT_HERSHEY_SIMPLEX

    # =====================================================
    # COMPANY NAME
    # =====================================================

    company_name = "JAY DEEPTHI INTERIORS"

    title_left = s(300)
    title_right = canvas_width - s(70)

    available_width = title_right - title_left

    company_scale = 1.0 * scale
    company_thickness = max(2, s(4))

    # Find largest size that fits
    while True:

        (
            text_width,
            text_height
        ), _ = cv2.getTextSize(
            company_name,
            font_bold,
            company_scale,
            company_thickness
        )

        if text_width >= available_width:
            company_scale -= 0.05 * scale
            break

        company_scale += 0.05 * scale

        if company_scale > 5:
            break

    (
        text_width,
        text_height
    ), _ = cv2.getTextSize(
        company_name,
        font_bold,
        company_scale,
        company_thickness
    )

    company_x = title_left + (
        available_width - text_width - 100
    ) // 2

    company_y = s(100)

    # =====================================================
    # SMALL GOLD ACCENT ABOVE COMPANY NAME
    # =====================================================

    accent_width = s(90)

    cv2.line(
        canvas,
        (
            company_x,
            company_y - s(28)
        ),
        (
            company_x + accent_width,
            company_y - s(28)
        ),
        GOLD,
        s(4),
        cv2.LINE_AA
    )

    # =====================================================
    # COMPANY NAME - MAROON
    # =====================================================

    cv2.putText(
        canvas,
        company_name,
        (company_x, company_y),
        font_bold,
        company_scale,
        MAROON,
        company_thickness,
        cv2.LINE_AA
    )

    # =====================================================
    # SUBTITLE
    # =====================================================

    subtitle = "INTERIOR DESIGN & DECORATION"

    subtitle_target_width = int(
        canvas_width * 0.65
    )

    subtitle_scale = 0.8 * scale
    subtitle_thickness = max(2, s(3))

    while True:

        (
            text_width,
            text_height
        ), _ = cv2.getTextSize(
            subtitle,
            font_bold,
            subtitle_scale,
            subtitle_thickness
        )

        if text_width >= subtitle_target_width:
            subtitle_scale -= 0.05 * scale
            break

        subtitle_scale += 0.05 * scale

        if subtitle_scale > 4:
            break

    (
        text_width,
        text_height
    ), _ = cv2.getTextSize(
        subtitle,
        font_bold,
        subtitle_scale,
        subtitle_thickness
    )

    subtitle_x = (
        canvas_width - text_width
    ) // 2

    subtitle_y = s(160)

    # =====================================================
    # GOLD DECORATIVE LINES
    # =====================================================

    line_gap = s(25)
    line_length = s(70)

    cv2.line(
        canvas,
        (
            subtitle_x - line_length - line_gap,
            subtitle_y - s(15)
        ),
        (
            subtitle_x - line_gap,
            subtitle_y - s(15)
        ),
        GOLD,
        s(3),
        cv2.LINE_AA
    )

    cv2.line(
        canvas,
        (
            subtitle_x + text_width + line_gap,
            subtitle_y - s(15)
        ),
        (
            subtitle_x + text_width +
            line_gap + line_length,
            subtitle_y - s(15)
        ),
        GOLD,
        s(3),
        cv2.LINE_AA
    )

    # =====================================================
    # SUBTITLE - DARK MAROON
    # =====================================================

    cv2.putText(
        canvas,
        subtitle,
        (subtitle_x, subtitle_y),
        font_bold,
        subtitle_scale,
        DARK_MAROON,
        subtitle_thickness,
        cv2.LINE_AA
    )

    # =====================================================
    # DIVIDER
    # =====================================================

    divider_y = s(190)

    cv2.line(
        canvas,
        (s(70), divider_y),
        (canvas_width - s(70), divider_y),
        LIGHT_GOLD,
        s(2),
        cv2.LINE_AA
    )

    # =====================================================
    # ADDRESS
    # =====================================================

    address = (
        "Nallapadu, Near HP Gas Booking Office, "
        "Guntur - 522005"
    )

    address_scale = 0.62 * scale
    address_thickness = max(2, s(2))

    address_target_width = int(
        canvas_width * 0.65
    )

    while True:

        (
            text_width,
            text_height
        ), _ = cv2.getTextSize(
            address,
            font_regular,
            address_scale,
            address_thickness
        )

        if text_width >= address_target_width:
            address_scale -= 0.04 * scale
            break

        address_scale += 0.04 * scale

        if address_scale > 3:
            break

    (
        text_width,
        text_height
    ), _ = cv2.getTextSize(
        address,
        font_regular,
        address_scale,
        address_thickness
    )

    address_x = (
        canvas_width - text_width
    ) // 2

    address_y = s(240)

    cv2.putText(
        canvas,
        address,
        (address_x, address_y),
        font_regular,
        address_scale,
        CHARCOAL,
        address_thickness,
        cv2.LINE_AA
    )

    # =====================================================
    # PHONE
    # =====================================================

    phone = "SURESH KUMAR, MOBILE : 99518 23287"

    phone_scale = 1.65 * scale
    phone_thickness = max(2, s(2))

    (
        text_width,
        text_height
    ), _ = cv2.getTextSize(
        phone,
        font_regular,
        phone_scale,
        phone_thickness
    )

    phone_x = (
        canvas_width - text_width
    ) // 2

    phone_y = s(300)

    cv2.putText(
        canvas,
        phone,
        (phone_x, phone_y),
        font_regular,
        phone_scale,
        DARK_MAROON,
        phone_thickness,
        cv2.LINE_AA
    )

    # =====================================================
    # SMALL GOLD PHONE ACCENT
    # =====================================================

    phone_line_width = s(180)

    cv2.line(
        canvas,
        (
            phone_x - phone_line_width - s(20),
            phone_y - s(8)
        ),
        (
            phone_x - s(20),
            phone_y - s(8)
        ),
        LIGHT_GOLD,
        s(2),
        cv2.LINE_AA
    )

    cv2.line(
        canvas,
        (
            phone_x + text_width + s(20),
            phone_y - s(8)
        ),
        (
            phone_x + text_width +
            phone_line_width + s(20),
            phone_y - s(8)
        ),
        LIGHT_GOLD,
        s(2),
        cv2.LINE_AA
    )

    # =====================================================
    # MAIN PREMIUM SEPARATOR
    # =====================================================

    separator_y = s(350)

    # Gold outer line
    cv2.line(
        canvas,
        (s(70), separator_y),
        (canvas_width - s(70), separator_y),
        GOLD,
        s(5),
        cv2.LINE_AA
    )

    # Thin maroon line underneath
    cv2.line(
        canvas,
        (
            s(70),
            separator_y + s(8)
        ),
        (
            canvas_width - s(70),
            separator_y + s(8)
        ),
        MAROON,
        s(2),
        cv2.LINE_AA
    )

    # =====================================================
    # SMALL CENTER DIAMOND
    # =====================================================

    center_x = canvas_width // 2
    center_y = separator_y

    diamond_size = s(10)

    diamond = np.array([
        [
            center_x,
            center_y - diamond_size
        ],
        [
            center_x + diamond_size,
            center_y
        ],
        [
            center_x,
            center_y + diamond_size
        ],
        [
            center_x - diamond_size,
            center_y
        ]
    ], dtype=np.int32)

    cv2.fillPoly(
        canvas,
        [diamond],
        MAROON
    )

    # =====================================================
    # PUT PROCESSED CONTENT UNDER LETTERHEAD
    # =====================================================

    canvas[
        header_height:
        header_height + content_height,
        0:content_width
    ] = content

    return canvas