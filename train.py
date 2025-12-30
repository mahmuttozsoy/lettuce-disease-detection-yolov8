from ultralytics import YOLO
from pathlib import Path
import multiprocessing as mp

def main():
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_YAML = BASE_DIR / "data" / "Lettuce-ObjDet-x3-2" / "data.yaml"

    model = YOLO("yolov8s.pt")  # detection

    model.train(
        data=str(DATASET_YAML),
        task="detect",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        patience=10,
        name="marul_detect_v2",
        project="runs"
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()
