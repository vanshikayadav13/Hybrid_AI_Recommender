"""
Image Utilities & Synthetic Image Generator Module

Manages local product image storage, image downloading, format conversions (RGB/RGBA/Grayscale),
and automatic synthetic image generation for offline testing & pipeline execution.
"""

from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import requests
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Category background color palettes for synthetic image generation
CATEGORY_COLORS = {
    "Electronics": (41, 128, 185),      # Deep Blue
    "Apparel": (142, 68, 173),          # Purple
    "Home & Kitchen": (39, 174, 96),    # Green
    "Books": (211, 84, 0),              # Orange/Brown
    "Beauty & Personal Care": (231, 76, 60)  # Coral Red
}


def create_synthetic_image(
    output_path: Path,
    product_name: str,
    category: str = "Electronics",
    size: Tuple[int, int] = (224, 224)
) -> Path:
    """
    Creates a distinct synthetic product image using Pillow.

    Args:
        output_path (Path): Target image filepath.
        product_name (str): Product title.
        category (str): Product category.
        size (Tuple[int, int]): Image dimensions.

    Returns:
        Path: Output image filepath.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Choose background color based on category
    bg_color = CATEGORY_COLORS.get(category, (52, 73, 94))
    image = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(image)

    # Draw visual shape based on product title hash for visual diversity
    name_hash = abs(hash(product_name))
    shape_type = name_hash % 3
    shape_color = (255, 255, 255, 180)

    if shape_type == 0:  # Circle
        draw.ellipse([40, 40, 184, 184], outline=shape_color, width=6)
    elif shape_type == 1:  # Rectangle
        draw.rectangle([50, 50, 174, 174], outline=shape_color, width=6)
    else:  # Polygon / Diamond
        draw.polygon([(112, 30), (194, 112), (112, 194), (30, 112)], outline=shape_color, width=6)

    # Add text label overlay
    short_title = product_name[:18] + "..." if len(product_name) > 18 else product_name
    draw.text((20, 100), short_title, fill=(255, 255, 255))
    draw.text((20, 125), f"[{category[:12]}]", fill=(240, 240, 240))

    image.save(output_path, "JPEG")
    return output_path


def ensure_catalog_images(
    products_df: pd.DataFrame,
    image_dir: Path = Path("data/images")
) -> Dict[str, Path]:
    """
    Ensures every product in products_df has a corresponding local image file.

    Args:
        products_df (pd.DataFrame): Products catalog dataset.
        image_dir (Path): Directory where images are stored.

    Returns:
        Dict[str, Path]: Dictionary mapping product_id -> local image Path.
    """
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    image_map = {}

    for _, row in products_df.iterrows():
        pid = str(row["product_id"]).strip().upper()
        pname = str(row.get("product_name", "Product"))
        cat = str(row.get("category", "General"))
        url = str(row.get("image_url", ""))

        local_path = image_dir / f"{pid.lower()}.jpg"

        # Check if local image already exists
        if local_path.exists():
            image_map[pid] = local_path
            continue

        # Try downloading if valid HTTP URL
        download_success = False
        if url.startswith("http://") or url.startswith("https://"):
            try:
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(resp.content)
                    download_success = True
                    logger.info("Downloaded product image for %s from %s", pid, url)
            except Exception as err:
                logger.warning("Failed downloading image for %s: %s. Generating synthetic fallback.", pid, err)

        # Fallback to synthetic image generation
        if not download_success:
            create_synthetic_image(local_path, product_name=pname, category=cat)
            logger.info("Generated synthetic image for product %s at %s", pid, local_path)

        image_map[pid] = local_path

    return image_map


def safe_load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Safely loads an image file and converts it to RGB mode.

    Args:
        image_path (Union[str, Path]): Path to image file.

    Returns:
        Image.Image: Loaded PIL Image in RGB format.

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image file is corrupt or unreadable.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        with Image.open(path) as img:
            # Convert RGBA / Grayscale / Palette to RGB
            rgb_img = img.convert("RGB")
            # Force load image bytes into memory before closing file handle
            rgb_img.load()
            return rgb_img
    except Exception as err:
        raise ValueError(f"Corrupt or invalid image file at {path}: {err}") from err
