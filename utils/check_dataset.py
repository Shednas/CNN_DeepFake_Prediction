"""Check dataset folder structure and image counts."""
from pathlib import Path

dataset_dir = Path(__file__).parent.parent / "dataset"  # ../dataset
subfolders = [
    dataset_dir / "train" / "real",
    dataset_dir / "train" / "ai_generated",
    dataset_dir / "test" / "real",
    dataset_dir / "test" / "ai_generated",
]

valid_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

for folder in subfolders:
    if folder.exists():
        images = [f for f in folder.glob('*') if f.suffix.lower() in valid_extensions]
        print(f"{folder.name:15} - {len(images):5} images")
    else:
        print(f"{folder.name:15} - FOLDER NOT FOUND")
