from pathlib import Path
import shutil

from ultralytics import YOLO

MODEL_PATH = Path("models/baseline/best.pt")
VAL_IMAGE_DIR = Path("data/raw/Dataset/images/val")

SAMPLE_DIR = Path("data/sample/inference_images")
OUTPUT_DIR = Path("results/predictions")


def create_sample_images(limit=20):
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list(VAL_IMAGE_DIR.glob("*.jpg"))[:limit]

    for image_file in image_files:
        shutil.copy2(image_file, SAMPLE_DIR / image_file.name)

    print(f"Copied {len(image_files)} sample images to {SAMPLE_DIR}")


def main():
    create_sample_images(limit=20)

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=str(SAMPLE_DIR),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="baseline_sample_predictions",
        exist_ok=True,
    )

    print("Inference completed successfully.")
    print(f"Predictions saved in: {OUTPUT_DIR / 'baseline_sample_predictions'}")


if __name__ == "__main__":
    main()