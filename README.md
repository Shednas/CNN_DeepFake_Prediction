# CNN DeepFake Prediction

A research-oriented deepfake detection system based on a custom **Convolutional Neural Network (CNN)** for classifying images as **real** or **AI-generated**. The project is designed for academic use, reproducibility, and clear methodological analysis rather than black-box performance optimization.

---

## Overview

This repository provides an end-to-end pipeline to:

* Train a CNN on real and AI-generated image datasets
* Evaluate performance using standard classification metrics
* Perform inference on unseen images
* Generate structured text reports and high-resolution visual dashboards

The system is suitable for dissertations, experimental benchmarking, and methodological studies in image forensics and deepfake detection.

---

## Model Architecture

The network is intentionally lightweight and interpretable:

```
Input: 128×128 RGB image
 → Conv(3→16) + ReLU + MaxPool
 → Conv(16→32) + ReLU + MaxPool
 → Conv(32→64) + ReLU + MaxPool
 → Flatten (16,384 features)
 → FC (128) + ReLU
 → FC (2 logits: ai_generated, real)
```

Training configuration:

* Optimizer: Adam
* Loss: CrossEntropyLoss
* Batch size: 32
* Epochs: 5
* Learning rate: 0.001

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Requirements:

* Python 3.8+
* PyTorch 2.7.1, torchvision
* OpenCV, NumPy, scikit-learn, Matplotlib, Pillow
* CUDA 11.8 (optional)

---

## Dataset Structure

```
CNNTestModel/dataset/
├── train/
│   ├── ai_generated/
│   └── real/
├── test/
│   ├── ai_generated/
│   └── real/
└── toBePredicted/
```

---

## Training

```bash
python CNNTestModel/script/train.py
```

Outputs:

* Trained model: `CNNTestModel/model/cnn_model.pth`
* Training history: `training_history.pkl`

---

## Testing

```bash
python CNNTestModel/script/test.py
```

Produces accuracy, per-class precision/recall/F1-score, and a confusion matrix on the test dataset.

Typical performance on a balanced dataset of 2,250 images:

* Accuracy: ~92–93%

---

## Inference (CLI)

Place image(s) in:

```
CNNTestModel/dataset/toBePredicted/
```

Run:

```bash
python CNNTestModel/visualization/deepfake_detector.py
```

Generated outputs (saved to `CNNTestModel/results/`):

* Text-based analysis report
* High-resolution (300 DPI) visualization dashboard

---

## Web Interface

A lightweight Streamlit application is provided for interactive use:

```bash
pip install streamlit
streamlit run CNNTestModel/visualization/streamlit_app.py
```

Features include image upload, real-time prediction, and downloadable reports.

---

## Project Structure

```
CNN_DeepFake_Prediction/
├── CNNTestModel/
│   ├── dataset/
│   ├── script/
│   ├── visualization/
│   ├── model/
│   ├── results/
│   ├── utils/
│   └── README.md
├── web_app/
├── requirements.txt
└── README.md
```

---

## Use Cases

* Academic research and dissertations
* CNN-based deepfake detection studies
* Image forensics experimentation
* Demonstrations and technical presentations

---

## Limitations and Future Work

This implementation focuses on CNN-based spatial features. Planned extensions include:

* Frequency-domain (FFT) feature integration
* Vision Transformer (ViT) models
* Hybrid and ensemble approaches
* Video-level deepfake detection

---

## License

MIT License

---

## Citation

If used in academic work, please cite:

```bibtex
@software{cnn_deepfake_2025,
  title={CNN DeepFake Prediction},
  author={Sandesh Thapa},
  year={2025},
  url={https://github.com/Shednas/CNN_DeepFake_Prediction}
}
```

---
**Last updated:** 18th December 2025
