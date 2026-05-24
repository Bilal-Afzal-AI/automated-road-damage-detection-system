import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# =========================
# CLASS MAPPING
# =========================
# RDD2022 classes converted to YOLO class IDs
CLASS_MAP = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3,  # Pothole
}

# =========================
# PATH CONFIGURATION
# =========================
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Your dataset structure:
# data/raw/train/images
# data/raw/train/labels
# data/raw/val/images
# data/raw/val/labels


def convert_box(image_width, image_height, xmin, ymin, xmax, ymax):
    """
    Convert Pascal VOC bounding box to YOLO format.
    """

    center_x = ((xmin + xmax) / 2) / image_width
    center_y = ((ymin + ymax) / 2) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return center_x, center_y, width, height


def convert_xml_file(xml_path, output_txt_path):
    """
    Convert one XML annotation file into one YOLO TXT label file.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text

        # Skip classes we are not using
        if class_name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[class_name]

        bndbox = obj.find("bndbox")

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        center_x, center_y, width, height = convert_box(
            image_width,
            image_height,
            xmin,
            ymin,
            xmax,
            ymax,
        )

        yolo_line = (
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        yolo_lines.append(yolo_line)

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))


def copy_images(split):
    """
    Copy images from raw dataset folder to processed dataset folder.
    """

    source_image_dir = RAW_DATA_DIR / split / "images"
    target_image_dir = PROCESSED_DATA_DIR / "images" / split

    target_image_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png"]

    copied_count = 0

    for image_path in source_image_dir.iterdir():
        if image_path.suffix.lower() in image_extensions:
            shutil.copy2(image_path, target_image_dir / image_path.name)
            copied_count += 1

    print(f"Copied {copied_count} images for {split}")


def convert_split(split):
    """
    Convert labels and copy images for one split.
    Example: train or val
    """

    source_label_dir = RAW_DATA_DIR / split / "labels"
    target_label_dir = PROCESSED_DATA_DIR / "labels" / split

    target_label_dir.mkdir(parents=True, exist_ok=True)

    xml_files = list(source_label_dir.glob("*.xml"))

    print(f"\nProcessing split: {split}")
    print(f"Found {len(xml_files)} XML label files")

    converted_count = 0

    for xml_path in xml_files:
        output_txt_path = target_label_dir / f"{xml_path.stem}.txt"
        convert_xml_file(xml_path, output_txt_path)
        converted_count += 1

    print(f"Converted {converted_count} label files for {split}")

    copy_images(split)


def main():
    """
    Main function to convert RDD2022 Pascal VOC labels to YOLO format.
    """

    print("Starting RDD2022 Pascal VOC to YOLO conversion...")

    for split in ["train", "val"]:
        convert_split(split)

    print("\nConversion completed successfully.")
    print("Processed dataset saved at: data/processed")


if __name__ == "__main__":
    main()