# 🥬 Lettuce Disease Detection (YOLOv8)

This project performs **object detection on lettuce leaf diseases**
using **YOLOv8**.

## 🚀 Features
- Multi-class object detection
- High accuracy (mAP@50 ≈ 97%)
- Tested on unseen test dataset
- Ready for real-world deployment

## 🧠 Detected Classes
- Bacterial
- Downy Mildew
- Powdery Mildew
- Septoria Blight
- Viral
- Wilt & Leaf Blight
- Healthy

## 📊 Test Results
| Metric | Value |
|------|------|
| Precision | 0.91 |
| Recall | 0.96 |
| mAP@50 | 0.97 |
| mAP@50-95 | 0.82 |

## 📂 Project Structure
```txt
lettuce_disease_detection/
├── data/
│   └── Lettuce-ObjDet-x3-2/
│       └── data.yaml
├── train.py
├── test_all.py
├── runs/ (ignored)
├── .gitignore
└── README.md

🛠 Training

python train.py

🔍 Evaluation

python test_all.py

📌 Notes

Dataset not included due to size and license.

You can download the dataset via Roboflow.


