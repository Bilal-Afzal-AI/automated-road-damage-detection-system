from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("models/baseline/best.pt")


def main():
    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=0,          # 0 = default webcam
        imgsz=640,
        conf=0.25,
        show=True,
        save=False,
    )


if __name__ == "__main__":
    main()