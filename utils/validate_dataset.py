"""Validate and clean dataset - remove corrupted images."""
from pathlib import Path
from PIL import Image

dataset_dir = Path(__file__).parent.parent / "dataset"  # ../dataset
valid_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

def check_image(img_path):
    """Try to open and convert image to RGB. Return True if valid."""
    try:
        img = Image.open(img_path)
        img.convert("RGB")  # Test conversion
        return True
    except Exception as e:
        print(f"  Corrupted: {img_path.name} - {type(e).__name__}")
        return False

# Check all folders
for folder_path in dataset_dir.rglob("*"):
    if folder_path.is_dir():
        images = [f for f in folder_path.glob("*") if f.suffix.lower() in valid_extensions]
        if images:
            print(f"\nValidating {folder_path.relative_to(dataset_dir)}...")
            removed = 0
            for img_path in images:
                if not check_image(img_path):
                    img_path.unlink()  # Delete corrupted file
                    removed += 1
            print(f"  Valid: {len(images) - removed}, Removed: {removed}")
