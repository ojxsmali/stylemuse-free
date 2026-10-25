"""No-API, lightweight hairstyle preview utilities for HairMuse.

Creates an illustrated hairstyle layer over a photo. This is deliberately NOT
photorealistic generative AI or a replacement for hairstyle transfer models.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import random

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


STYLES = (
    "Short Crop", "Side Part", "Curly", "Wavy Bob",
    "Long Layers", "Buzz Cut", "Pompadour", "Ponytail",
)

COLORS = {
    "Natural Black": "#242024",
    "Dark Brown": "#503128",
    "Chocolate": "#71452e",
    "Copper": "#bc603c",
    "Ash Blonde": "#bfa574",
    "Platinum": "#ded3be",
    "Burgundy": "#70334a",
    "Pastel Pink": "#cc819e",
    "Midnight Blue": "#324c79",
}


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    detected: bool


def prepare_image(image: Image.Image, max_size: int = 1400) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def detect_face(image: Image.Image) -> FaceBox:
    """Find biggest roughly frontal face; fall back to a centered estimate."""
    w, h = image.size
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if not cascade.empty():
            rgb = np.asarray(image)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(max(30, w // 12), max(30, h // 12)),
            )
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda r: int(r[2]) * int(r[3]))
                return FaceBox(int(x), int(y), int(fw), int(fh), True)
    except (cv2.error, AttributeError, ValueError):
        pass
    # Centered portrait assumption: easy to correct with the adjustment sliders.
    side = max(1, int(min(w * 0.53, h * 0.42)))
    return FaceBox(
        x=(w - side) // 2,
        y=max(0, int(h * 0.23)),
        width=side,
        height=side,
        detected=False,
    )


def _cap(draw: ImageDraw.ImageDraw, *, high: bool = False) -> None:
    if high:
        draw.polygon([
            (100, 265), (108, 159), (135, 86), (182, 44),
            (224, 37), (247, 5), (290, 28), (334, 1),
            (381, 50), (426, 40), (469, 102), (500, 190),
            (508, 265), (469, 290), (453, 225), (401, 211),
            (360, 227), (320, 206), (266, 235), (218, 220),
            (157, 276),
        ], fill=255)
    else:
        draw.polygon([
            (99, 261), (103, 181), (143, 97), (217, 64),
            (307, 53), (391, 65), (455, 96), (496, 176),
            (503, 262), (465, 286), (446, 221), (401, 223),
            (356, 232), (311, 217), (264, 243), (206, 223),
            (153, 282),
        ], fill=255)


def make_hair_layer(style: str, color: str) -> Image.Image:
    if style not in STYLES:
        raise ValueError("Unknown hairstyle")
    if color not in COLORS.values():
        raise ValueError("Unknown hair color")
    canvas_w, canvas_h = 600, 650
    mask = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(mask)

    if style == "Buzz Cut":
        draw.ellipse((136, 115, 464, 293), fill=255)
        draw.polygon([(139, 221), (152, 262), (202, 244), (278, 234),
                      (351, 248), (424, 232), (461, 253), (465, 213)], fill=255)
    elif style == "Short Crop":
        _cap(draw)
        draw.polygon([(153, 217), (195, 267), (243, 251), (279, 279),
                      (315, 251), (352, 260), (417, 221)], fill=255)
        draw.rounded_rectangle((107, 228, 153, 346), 18, fill=255)
        draw.rounded_rectangle((449, 228, 491, 343), 18, fill=255)
    elif style == "Side Part":
        _cap(draw)
        draw.polygon([(145, 157), (201, 233), (275, 258), (346, 266),
                      (431, 232), (465, 176), (415, 128), (295, 156)], fill=255)
        draw.rounded_rectangle((106, 240, 154, 359), 18, fill=255)
        draw.rounded_rectangle((448, 240, 493, 356), 18, fill=255)
    elif style == "Curly":
        _cap(draw)
        rng = random.Random(92)
        for row, yy in enumerate((108, 148, 186, 222)):
            for col in range(8):
                xx = 122 + col * 48 + (16 if row % 2 else 0)
                xx += rng.randint(-14, 12)
                ry = rng.randint(31, 53)
                rx = rng.randint(33, 50)
                draw.ellipse((xx-rx, yy-ry, xx+rx, yy+ry), fill=255)
        for xx, bottom in ((101, 391), (474, 398)):
            for yy in range(251, bottom, 45):
                draw.ellipse((xx - 31, yy - 35, xx + 39, yy + 38), fill=255)
    elif style == "Wavy Bob":
        draw.polygon([(105, 215), (162, 217), (180, 406), (160, 476),
                      (176, 509), (135, 538), (87, 501), (102, 449),
                      (82, 388), (100, 330)], fill=255)
        draw.polygon([(439, 217), (499, 214), (508, 338), (530, 396),
                      (508, 457), (525, 502), (466, 541), (430, 505),
                      (450, 460), (429, 397)], fill=255)
        _cap(draw)
        draw.arc((115, 130, 485, 370), 188, 340, fill=255, width=29)
    elif style == "Long Layers":
        draw.polygon([(107, 220), (165, 213), (170, 361), (164, 475),
                      (183, 583), (170, 644), (109, 628), (85, 542),
                      (98, 440), (79, 346)], fill=255)
        draw.polygon([(439, 217), (493, 215), (522, 350), (502, 446),
                      (523, 547), (489, 648), (429, 637), (434, 575),
                      (442, 478), (425, 359)], fill=255)
        _cap(draw)
        draw.polygon([(160, 207), (221, 246), (301, 227), (397, 218),
                      (449, 227), (413, 163), (233, 170)], fill=255)
    elif style == "Pompadour":
        _cap(draw, high=True)
        draw.ellipse((175, 0, 423, 191), fill=255)
        draw.polygon([(170, 154), (232, 229), (314, 230), (412, 195),
                      (467, 164), (415, 95), (229, 95)], fill=255)
        draw.rounded_rectangle((108, 221, 154, 353), 19, fill=255)
        draw.rounded_rectangle((447, 221, 493, 352), 19, fill=255)
    elif style == "Ponytail":
        draw.ellipse((452, 171, 548, 294), fill=255)
        draw.polygon([(490, 236), (551, 256), (565, 361), (553, 467),
                      (585, 588), (531, 619), (507, 548), (510, 455),
                      (475, 338)], fill=255)
        _cap(draw)
        draw.polygon([(139, 183), (243, 242), (343, 247), (454, 199),
                      (428, 139), (222, 153)], fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.5))
    base = Image.new("RGBA", (canvas_w, canvas_h), color)
    base.putalpha(mask)

    # Subtle illustrated strands, clipped to the hair silhouette.
    details = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(details)
    rng = random.Random(11)
    for i in range(23):
        x = 90 + i * 18 + rng.randint(-8, 8)
        y = rng.randint(46, 96)
        d.arc((x - 90, y, x + 95, y + 240), 190, 297,
              fill=(255, 255, 255, rng.randint(18, 48)), width=rng.randint(2, 4))
    if style in ("Long Layers", "Wavy Bob", "Ponytail"):
        for x in list(range(96, 162, 15)) + list(range(447, 515, 15)):
            d.line([(x, 280), (x + rng.randrange(-15, 16), 445),
                    (x + rng.randrange(-25, 26), 609)],
                   fill=(255, 255, 255, 25), width=3)
    alpha = ImageChops.multiply(details.getchannel("A"), mask)
    details.putalpha(alpha)
    return Image.alpha_composite(base, details)


def render_preview(
    image: Image.Image,
    style: str,
    color: str,
    face: FaceBox,
    offset_x: float = 0,
    offset_y: float = 0,
    size_percent: int = 100,
) -> Image.Image:
    """Place illustrated hair on a photo; offsets are percentages of face size."""
    if not 50 <= size_percent <= 200:
        raise ValueError("size_percent must be between 50 and 200")
    image = image.convert("RGBA")
    width = max(20, int(face.width * 1.65 * size_percent / 100))
    height = max(20, int(face.height * 1.75 * size_percent / 100))
    hair = make_hair_layer(style, color).resize((width, height), Image.Resampling.LANCZOS)
    cx = face.x + face.width / 2 + face.width * offset_x / 100
    top = face.y - height * 0.35 + face.height * offset_y / 100
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer.paste(hair, (int(cx - width / 2), int(top)))
    return Image.alpha_composite(image, layer).convert("RGB")


def to_png_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def sample_portrait() -> Image.Image:
    """A built-in drawing, so no external sample photo or upload is required."""
    image = Image.new("RGB", (720, 900), "#eee0dc")
    d = ImageDraw.Draw(image)
    d.ellipse((-180, -170, 380, 500), fill="#f1d0c9")
    d.ellipse((460, 400, 1020, 1060), fill="#e8c7c1")
    d.polygon([(130, 900), (205, 660), (300, 617), (424, 618),
               (515, 665), (585, 900)], fill="#72668f")
    d.rounded_rectangle((326, 618, 400, 717), radius=27, fill="#b8745c")
    d.ellipse((214, 172, 506, 688), fill="#c98d72")
    d.ellipse((212, 379, 245, 461), fill="#c98d72")
    d.ellipse((483, 379, 515, 462), fill="#c98d72")
    d.arc((250, 306, 343, 395), start=205, end=320, fill="#684434", width=7)
    d.arc((378, 306, 471, 395), start=205, end=320, fill="#684434", width=7)
    d.ellipse((291, 373, 306, 383), fill="#3a2c2c")
    d.ellipse((416, 373, 431, 383), fill="#3a2c2c")
    d.line((360, 375, 345, 473, 371, 488), fill="#ad6758", width=5)
    d.arc((306, 489, 418, 566), start=18, end=164, fill="#874d4c", width=7)
    d.polygon([(208, 379), (217, 247), (266, 168), (357, 135),
               (452, 176), (500, 284), (502, 367), (451, 240),
               (417, 223), (359, 255), (269, 253)], fill="#403031")
    return image
