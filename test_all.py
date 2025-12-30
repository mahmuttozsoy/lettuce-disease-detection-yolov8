from ultralytics import YOLO
from pathlib import Path
import multiprocessing as mp

def main():
    MODEL_PATH = Path("runs/marul_detect_v2/weights/best.pt")
    TEST_IMAGES = Path("data/Lettuce-ObjDet-x3-2/test/images")
    DATA_YAML = Path("data/Lettuce-ObjDet-x3-2/data.yaml")

    print("Model exists:", MODEL_PATH.exists())
    print("Test images exists:", TEST_IMAGES.exists())
    print("Data yaml exists:", DATA_YAML.exists())

    model = YOLO(MODEL_PATH)

    # 🔹 Toplu inference
    model.predict(
        source=str(TEST_IMAGES),
        conf=0.25,
        iou=0.5,
        save=True,
        save_txt=True,
        imgsz=640,
        device=0,
        workers=0   # 🔑 Windows için kritik
    )

    print("✅ Tüm test dataseti inference tamamlandı")

    # 🔹 Test değerlendirme
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        conf=0.25,
        iou=0.5,
        device=0,
        workers=0   # 🔑 yine kritik
    )

    print("📊 Test metrikleri:")
    print(metrics)
    print("✅ Test süreci tamamen tamamlandı")

if __name__ == "__main__":
    mp.freeze_support()   # 🔑 Windows spawn fix
    main()
