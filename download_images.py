"""
download_images.py

Downloads Alps/mountain landscape images from Wikimedia Commons using their public API.
Images are freely licensed (Creative Commons) and saved with proper attribution tracking.

Usage:
    python download_images.py --num-images 300 --output-dir data/raw-images
"""

import os
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Wikimedia Commons API endpoint
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
            "User-Agent": "MLCrumblingAlps/1.0 (https://github.com/Mass23/ML_crumblingalps; massimo.bourquin@gmail.com) requests/2.31.0"
        }
# Search queries to find diverse Alps/mountain images
SEARCH_QUERIES = [
    "Alps mountains landscape",
    "Swiss Alps panorama",
    "Alpine landscape",
    "Dolomites mountains",
    "Mont Blanc landscape",
    "Austrian Alps",
    "Italian Alps",
    "French Alps",
    "Swiss Alps",
    "Switzerland mountains landscape",
    "Switzerland Alpage",
    "Montagnes suisses",
    "Alpage suisse",
    "Chalet montagnes suisses",
    "Mountain panorama landscape",
    "Alps snow peaks",
    "Alpine valley landscape",
    "Switzerland Alps",
]

EXCLUDE_KEYWORDS = ["New Zealand", "new zealand", "Zealand", "NZ_Southern_Alp", "NZ", "new_zealand", 'New_Zealand',
                    "New zeal", "new zeal", "New_zeal", "new_zeal", 
                    "Norway", "norway", "norwegian alps", "Norwegian_Alps", "norwegian_alps"]

def is_valid_filename(filename):
    filename_lower = filename.lower()
    return not any(keyword in filename_lower for keyword in EXCLUDE_KEYWORDS)

# Rate limiting: 1 second between requests
REQUEST_DELAY = 1.0


def search_wikimedia_images(query: str, limit: int = 50) -> list[dict]:
    """
    Search Wikimedia Commons for images matching the given query.

    Args:
        query: Search query string
        limit: Maximum number of results to return per query (max 50)

    Returns:
        List of dicts with image metadata
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",  # File namespace
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": 1024,
    }

    try:
        response = requests.get(WIKIMEDIA_API, params=params, timeout=30, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        filtered_pages = {
            page_id: page
            for page_id, page in pages.items()
            if is_valid_filename(page.get("title", ""))
        }
        return list(filtered_pages.values())
    except Exception as e:
        print(f"  Error searching for '{query}': {e}")
        return []


def extract_image_info(page: dict) -> Optional[dict]:
    """
    Extract relevant image information from a Wikimedia API page result.

    Args:
        page: Page dict from Wikimedia API response

    Returns:
        Dict with image metadata, or None if the image is not suitable
    """
    imageinfo_list = page.get("imageinfo", [])
    if not imageinfo_list:
        return None

    info = imageinfo_list[0]
    url = info.get("url", "")
    thumb_url = info.get("thumburl", url)

    # Only download images at least 1024px wide
    width = info.get("width", 0)
    height = info.get("height", 0)
    if width < 1024:
        return None

    # Extract license and attribution from extended metadata
    extmeta = info.get("extmetadata", {})
    license_short = extmeta.get("LicenseShortName", {}).get("value", "")
    author = extmeta.get("Artist", {}).get("value", "Unknown")
    license_url = extmeta.get("LicenseUrl", {}).get("value", "")

    # Filter for Creative Commons licenses
    if not any(cc in license_short for cc in ["CC BY", "CC0", "Public domain", "CC-BY"]):
        return None

    title = page.get("title", "").replace("File:", "").strip()

    # Sanitize filename: replace spaces and special characters
    safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)

    # Determine extension from URL or title
    ext = Path(url).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif"]:
        ext = ".jpg"

    filename = f"{safe_title}{ext}" if not safe_title.endswith(ext) else safe_title

    return {
        "filename": filename,
        "title": title,
        "download_url": thumb_url if thumb_url else url,
        "source_url": f"https://commons.wikimedia.org/wiki/File:{title.replace(' ', '_')}",
        "author": author,
        "license": license_short,
        "license_url": license_url,
        "width": width,
        "height": height,
    }


def download_image(image_info: dict, output_dir: Path) -> bool:
    """
    Download a single image to the output directory.

    Args:
        image_info: Dict with image metadata including download_url and filename
        output_dir: Directory to save the image

    Returns:
        True if successful, False otherwise
    """
    filename = image_info["filename"]
    output_path = output_dir / filename

    # Skip if already downloaded
    if output_path.exists():
        print(f"  Skipping (already exists): {filename}")
        return True

    try:
        response = requests.get(image_info["download_url"], headers=HEADERS, timeout=60, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        # Clean up partial download
        if output_path.exists():
            output_path.unlink()
        return False


def load_attributions(attr_path: Path) -> dict:
    """Load existing attributions JSON, or return an empty dict."""
    if attr_path.exists():
        with open(attr_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_attributions(attributions: dict, attr_path: Path) -> None:
    """Save attributions dict to JSON file."""
    with open(attr_path, "w", encoding="utf-8") as f:
        json.dump(attributions, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Alps/mountain images from Wikimedia Commons"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=300,
        help="Number of images to download (default: 300)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw-images",
        help="Directory to save downloaded images (default: data/raw-images)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attr_path = output_dir.parent / "attributions.json"
    attributions = load_attributions(attr_path)

    downloaded = 0
    target = args.num_images

    print(f"Downloading up to {target} images to '{output_dir}'...")
    print(f"Attributions will be saved to '{attr_path}'")

    for query in SEARCH_QUERIES:
        if downloaded >= target:
            break

        remaining = target - downloaded
        per_query = min(50, remaining)

        print(f"\nSearching: '{query}' (need {remaining} more images)...")
        time.sleep(REQUEST_DELAY)

        pages = search_wikimedia_images(query, limit=per_query)
        print(f"  Found {len(pages)} candidate pages")

        for page in pages:
            if downloaded >= target:
                break

            image_info = extract_image_info(page)
            if image_info is None:
                continue

            filename = image_info["filename"]

            # Skip if attribution already recorded (already downloaded previously)
            if filename in attributions and (output_dir / filename).exists():
                print(f"  Skipping (already tracked): {filename}")
                downloaded += 1
                continue

            print(f"  Downloading [{downloaded + 1}/{target}]: {filename}")

            time.sleep(REQUEST_DELAY)
            success = download_image(image_info, output_dir)

            if success:
                attributions[filename] = {
                    "filename": filename,
                    "title": image_info["title"],
                    "source_url": image_info["source_url"],
                    "author": image_info["author"],
                    "license": image_info["license"],
                    "license_url": image_info["license_url"],
                    "width": image_info["width"],
                    "height": image_info["height"],
                }
                save_attributions(attributions, attr_path)
                downloaded += 1

    print(f"\nDone! Downloaded {downloaded} images to '{output_dir}'")
    print(f"Attribution data saved to '{attr_path}'")


def get_batch_of_images(output_dir: str, num_images: int = 5, start_index: int = 0) -> list[str]:
    """
    Return a batch of image file paths from an existing directory.

    Args:
        output_dir: Directory containing downloaded images
        num_images: Number of images to include in the batch
        start_index: Starting index into the sorted list of images

    Returns:
        List of absolute file paths for the batch (may be shorter than
        num_images if fewer images remain after start_index)
    """
    output_path = Path(output_dir)
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]

    filepaths: list[str] = []
    for pattern in patterns:
        filepaths.extend(str(p) for p in output_path.glob(pattern))
    filepaths = sorted(set(filepaths))

    return filepaths[start_index : start_index + num_images]


if __name__ == "__main__":
    main()
