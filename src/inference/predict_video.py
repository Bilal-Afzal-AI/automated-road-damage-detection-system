from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("models/baseline/best.pt")
VIDEO_SOURCE = Path("data/sample/test_video.mp4")
OUTPUT_DIR = Path("results/video_predictions")


def main():
    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=str(VIDEO_SOURCE),
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="baseline_video_prediction",
        exist_ok=True,
    )

    print("Video inference completed.")
    print(f"Saved output in: {OUTPUT_DIR / 'baseline_video_prediction'}")


if __name__ == "__main__":
    main()