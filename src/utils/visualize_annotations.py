import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


DATASET_DIR = Path("data/raw/Dataset")

IMAGE_DIR = DATASET_DIR / "images" / "train"
LABEL_DIR = DATASET_DIR / "labels" / "train"

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
    4: "Unknown_Damage",
}


def draw_yolo_boxes(image_path, label_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape

    if label_path.exists():
        lines = label_path.read_text().strip().splitlines()

        for line in lines:
            parts = line.split()
            class_id = int(parts[0])

            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            label = CLASS_NAMES.get(class_id, "Unknown")

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    return image


def main():
    image_files = list(IMAGE_DIR.glob("*.jpg"))

    sample_images = random.sample(image_files, 6)

    for image_path in sample_images:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        image = draw_yolo_boxes(image_path, label_path)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(image_path.name)
        plt.show()


if __name__ == "__main__":
    main()