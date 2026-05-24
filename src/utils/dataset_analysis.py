from pathlib import Path
from collections import Counter

DATASET_DIR = Path("data/raw/Dataset")
LABELS_DIR = DATASET_DIR / "labels"

CLASS_NAMES = {
    "0": "D00_Longitudinal_Crack",
    "1": "D10_Transverse_Crack",
    "2": "D20_Alligator_Crack",
    "3": "Unknown_Damage",
}


def analyze_split(split):
    label_dir = LABELS_DIR / split
    label_files = list(label_dir.glob("*.txt"))

    class_counter = Counter()
    empty_files = 0
    total_boxes = 0

    for label_file in label_files:
        content = label_file.read_text().strip()

        if not content:
            empty_files += 1
            continue

        for line in content.splitlines():
            class_id = line.split()[0]
            class_counter[class_id] += 1
            total_boxes += 1

    print(f"\n===== {split.upper()} SET =====")
    print(f"Label files: {len(label_files)}")
    print(f"Empty label files: {empty_files}")
    print(f"Total bounding boxes: {total_boxes}")

    print("\nClass distribution:")
    for class_id, count in sorted(class_counter.items()):
        print(f"{class_id} - {CLASS_NAMES.get(class_id, 'Unknown')}: {count}")


def main():
    print("RDD2022 Dataset Analysis")
    analyze_split("train")
    analyze_split("val")


if __name__ == "__main__":
    main()