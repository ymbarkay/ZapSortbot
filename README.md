# ZapSortBot

**Autonomous Robotic Sorting of Used Batteries Using Vision, Weight Sensing, and Robotics**

ZapSortBot is an open-source research prototype for **autonomous battery sorting**, combining **computer vision**, **weight-based inference**, and **robotic manipulation**.
It demonstrates a **low-cost, modular alternative** to industrial battery-sorting systems that rely on spectroscopy or X-ray imaging.

This repository accompanies the peer-reviewed paper preprint:

> **ZapSortBot: Autonomous robotic sorting of used batteries utilizing deep learning**
> Y. Barkay Molimi *et al.*, IEEE IECON
>

---

## 🚀 Key Features

* **Multi-modal classification**

  * YOLOv8-based vision for battery detection
  * Weight-based inference for chemistry identification
  * Color + size heuristics for robustness and safety
* **Robotic manipulation**

  * 6-DoF Niryo Ned 2 robotic arm
  * Deterministic pick–lift–weigh–drop pipeline
* **Edge-case safety handling**

  * Ambiguous batteries are flagged as *unknown*
  * Prevents hazardous misclassification
* **Low-cost & accessible**

  * Uses standard RGB camera + load cell
  * Designed for decentralized recycling environments
* **Open research architecture**

  * Modular Python code
  * Reproducible training and inference pipeline

---

## 🧠 System Overview

**Pipeline:**

1. Battery detected on conveyor using YOLOv8
2. Visual features extracted (size, color, bounding box)
3. Battery picked and weighed
4. Chemistry inferred via rule-based fusion logic
5. Robot deposits battery into designated bin

**Supported chemistries (prototype):**

* Alkaline
* Zinc-carbon
* Nickel–Metal Hydride (NiMH)
* Lithium

---

## 📁 Repository Structure

```text
ZapSortBot/
├── train.py                  # YOLOv8 training script
├── robot_classification.py   # Main robotic control + GUI
├── battery_detector.py       # Vision-based detection logic
├── weight.py                 # ESP32 load-cell interface
├── classes.yaml              # YOLO dataset configuration
├── main.py                   # Main UI
├── best.pt                   # Best Yolo Model
├── datasets/                 # Training images & annotations
├── runs/                     # Trained YOLO weights
├── paper/                    # Conference paper & figures
├── README.md
└── LICENSE
```

---

## 🧪 Model Training

The YOLOv8 detector is fine-tuned from a COCO-pretrained backbone using a custom dataset of AA and AAA batteries.

Example training command (from `train.py`):

```python
model.train(
    data="classes.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    optimizer="SGD",
    lr0=0.005,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    scale=0.5,
)
```

Validation performance:

* **Precision:** 99.76%
* **Recall:** 100%
* **mAP@0.5:** 99.5%
* **Overall system accuracy:** 98.3% 

---

## 🤖 Robotic Control

* Robot: **Niryo Ned 2**
* Communication: Python API (`pyniryo`)
* Motion: Predefined joint-space waypoints
* GUI: Built with **Flet** for live monitoring and control

The system is designed for **deterministic, explainable motion**, not reinforcement learning.

---

## 🔬 Research Scope & Disclaimer

> **Important:**
> This repository represents a **research prototype** developed for academic validation.

* It is **not an industrial product**
* It is **not optimized for throughput**
* It is **not intended for direct commercial deployment**

Any future commercial system will:

* use a **different sensing architecture**
* employ **new hardware layouts**
* implement **independent inference algorithms**
* reuse **no code, models, datasets, or designs** from this repository

This repository exists **solely for research, education, and reproducibility**.

---

## 📜 License (Open Source)

This project is released under the **MIT License**.

* ✔ Free to use
* ✔ Free to modify
* ✔ Free for commercial and non-commercial use
* ✔ Attribution required
* ❌ No warranty

See `LICENSE` below.

---

## 📚 Citation

If you use this work in research, please cite:

```bibtex
@inproceedings{barkay2025zapsortbot,
  title={ZapSortBot: Autonomous robotic sorting of used batteries utilizing deep learning},
  author={Barkay Molimi, Youssouf and Pottmaier, Daphiny and Mayers, Jarede and Khalid, Syed Ghufran and Navaraj, William},
  booktitle={IEEE IECON},
  year={2025}
}
```

---

## 👤 Author

**Youssouf Barkay Molimi**
Email: [ymbarkay@gmail.com](mailto:ymbarkay@gmail.com)

Academic supervision and publication support acknowledged in the accompanying paper.

---

## 🔮 Future Work (Research)

* Additional sensing modalities (thermal, spectroscopy)
* Faster mechanical actuation
* Closed-loop robotic feedback
* Expanded battery form factors

---

# LICENSE (MIT)

```text
MIT License

Copyright (c) 2025 Youssouf Barkay Molimi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

