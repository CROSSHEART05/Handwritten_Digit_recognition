# ⚡ DigitMind: Multi-Digit Handwritten Recognition & Cybernetic UI

> **An end-to-end Computer Vision & Deep Learning system capable of detecting single and multi-digit handwritten numbers (e.g. `427`, `9012`) written on an interactive, attention-seeking canvas.**
> 
> *Fully optimized for **VS Code** and **Google Colab**.*

---

## 🌟 Key Features

1. **Multi-Digit Handwriting Detection**: 
   - Write single digits or entire multi-digit numbers side-by-side on the screen (e.g., `582`, `1094`).
   - Automated contour segmentation isolates every digit, extracts bounding boxes, and reads left-to-right.
2. **Deep Learning Engine (PyTorch CNN)**:
   - High-performance 2D Convolutional Neural Network with Batch Normalization, Dropout, and Max Pooling.
   - Evaluated to **>99.2% accuracy** on the MNIST dataset.
3. **Attention-Seeking Cybernetic UI**:
   - Cyber-dark aesthetic featuring glowing neon brush tracks (`#38bdf8`), real-time stroke smoothing, and sleek control panels.
   - Dynamic Prediction HUD: Interactive breakdown for each individual digit with confidence percentage bars and OpenCV bounding box overlays.
4. **VS Code & Google Colab Native**:
   - Run as a standalone web app in VS Code (`python app.py`).
   - Open as a notebook directly in VS Code Jupyter Extension or Google Colab.

---

## 🚀 How to Run in VS Code

### Option 1: Standalone Web App (Recommended for VS Code)
1. Open the project folder in VS Code (`File > Open Folder... > d:\Project\digit_recognition_notebook`).
2. Open the built-in VS Code Terminal (`Ctrl + ~` or `Terminal > New Terminal`).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Click the link `http://127.0.0.1:7860` to open your interactive cyber-dark drawing canvas in your browser!

### Option 2: Native VS Code Notebook
1. Open `mnist_digit_recognition.ipynb` in VS Code.
2. Click **Select Kernel** (top right) and choose your Python environment.
3. Click **Run All** at the top of the notebook.

---

## ☁️ How to Run in Google Colab

1. Go to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `mnist_digit_recognition.ipynb` (or upload `app.py` and `train_model.py`).
3. In Colab, run:
   ```python
   !pip install -r requirements.txt
   !python app.py
   ```
4. Click the generated `.gradio.live` link to open the interactive canvas on any device or mobile screen!

---

## 📁 Repository Structure

```
digit_recognition_notebook/
├── app.py                           # Standalone Web App (Best for VS Code & Colab)
├── mnist_digit_recognition.ipynb    # Interactive Python Notebook
├── train_model.py                   # PyTorch CNN definition & OpenCV segmentation engine
├── mnist_cnn.pth                    # Saved PyTorch CNN weights
├── requirements.txt                 # Dependencies list
└── README.md                        # Project documentation (this file)
```
