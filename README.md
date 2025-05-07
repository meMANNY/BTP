
# Tongue Segmentation Project

This repository implements a complete pipeline for segmenting a human tongue into specific anatomical regions (heart, liver, stomach) using:

* **Segment Anything Model (SAM)**: For rapid mask generation from polygon annotations.
* **U-Net** and **SegNet**: Deep learning architectures trained on the generated masks for fully automatic inference.

---

## 📂 Directory Structure

```
project_root/
├── checkpoints/           # SAM checkpoints & model weights
│   └── sam_vit_h.pth      # Downloaded SAM ViT-H model
│   └── unet_best.h5       # Best U-Net checkpoint
│   └── segnet_best.h5     # Best SegNet checkpoint
│
├── Dataset/               # Input data and generated masks
│   ├── images/            # Raw tongue images (.jpg/.png)
│   ├── annots/            # LabelMe JSON annotations
│   └── masks/             # Raw label masks (0=BG,1–3=regions)
│
├── outputs/               # SAM-generated and color previews
│   ├── masks/             # SAM raw masks
│   └── color_masks/       # Colored SAM mask previews
│
├── Models/                # Training artifacts and logs
│   ├── checkpoints/       # Saved U-Net & SegNet weights
│   └── logs/              # TensorBoard logs
│
├── utils.py               # Data loader, preprocessing, metrics
├── train_unet.py          # U-Net training script
├── train_segnet.py        # SegNet training script
└── run_sam_masks.py       # SAM mask generation script
```

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/meMANNY/BTP.git
   cd BTP
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python3 -m venv env
   source env/bin/activate   # macOS/Linux
   env\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🗂 Data Preparation

1. **Place your raw images** in `Dataset/images/`.
2. **Place your LabelMe JSON annotations** in `Dataset/annots/` (filename.json matches filename.jpg).
3. **Generate SAM masks** (run `run_sam_masks.py`) to produce:

   * `outputs/masks/` (raw mask arrays)
   * `outputs/color_masks/` (RGB previews)

---

## 🖼️ SAM Mask Generation

```bash
python run_sam_masks.py
```

* Uses `checkpoints/sam_vit_h.pth` to load the SAM ViT-H model.
* Reads each JSON polygon, computes centroid prompts, and generates region masks.
* Saves raw masks and color-coded previews.

---

## 🏋️ U-Net Training

```bash
python train_unet.py
```

* Reads `Dataset/images/` and `Dataset/masks/`.
* Splits data into training (80%) and validation (20%).
* Trains a U-Net on 256×256 images for 30 epochs.
* Saves best weights to `Models/checkpoints/unet_best.h5`.
* Logs to `Models/logs/unet` for TensorBoard.

Launch TensorBoard:

```bash
tensorboard --logdir Models/logs/unet
```

---

## 🏋️ SegNet Training

```bash
python train_segnet.py
```

* Same data pipeline as U-Net.
* Trains a SegNet architecture for 30 epochs.
* Saves best weights to `Models/checkpoints/segnet_best.h5`.
* Logs to `Models/logs/segnet` for TensorBoard.

Launch TensorBoard:

```bash
tensorboard --logdir Models/logs/segnet
```

---

## 🎯 Inference

After training, load either model and run inference on new images:

```python
from tensorflow.keras.models import load_model
import cv2, numpy as np

# Load model
model = load_model('Models/checkpoints/unet_best.h5')  # or segnet_best.h5

# Read & preprocess
img = cv2.imread('path/to/new_image.jpg')
img = cv2.resize(img, (256,256))/255.0
pred = model.predict(np.expand_dims(img,0))[0]
mask = np.argmax(pred, axis=-1).astype(np.uint8)

# Save or display mask
cv2.imwrite('pred_mask.png', mask*80)
```

For full inference examples, refer to the scripts in this repo.

---

## 🔧 Tips & Notes

* **Adjust IMG\_SIZE** in `utils.py` and training scripts for higher resolution if you have GPU memory.
* **Augment data** (flips, rotations) by extending `utils.py` `data_generator` for robustness.
* **Early stopping** and **learning-rate schedules** can improve training.
* **Class imbalance**: consider weighted loss or focal loss if some regions are underrepresented.

---

## 📚 References

* [Segment Anything Model (SAM) GitHub](https://github.com/facebookresearch/segment-anything)
* [U-Net: Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
* [SegNet: Badrinarayanan et al., 2017](https://arxiv.org/abs/1511.00561)

---

Happy segmenting! 🎉

---
