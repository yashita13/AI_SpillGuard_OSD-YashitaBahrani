# 🌊 AI SpillGuard: An AI-Powered System for Oil Spill Identification and Monitoring

This repository contains the source code and documentation for **AI SpillGuard**, a deep learning-powered web application designed for the automated detection and segmentation of oil spills from satellite imagery. This project was developed as part of the Infosys AI internship.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)

---

## 🚀 Deployed Application

You can access and test the live application here:

**[https://yashita-oilspilldetector1234554321.streamlit.app/](https://yashita-oilspilldetector1234554321.streamlit.app/)**

---

## 📖 Project Overview

Oil spills pose a severe threat to marine ecosystems, coastal regions, and local economies. Traditional detection methods are often slow, labor-intensive, and reactive. This project aims to solve this problem by leveraging a state-of-the-art deep learning model to identify and localize oil spills efficiently and accurately from satellite imagery.

The final result is an interactive web application where a user can upload a satellite image and receive a detailed analysis, including a visual segmentation map and quantitative statistics about the detected spill.

## ✨ Key Features

- **Multi-Class Segmentation:** The model doesn't just find spills; it distinguishes between **Oil**, **Water**, **Land/Other**, and **Background**, leading to highly accurate and reliable predictions.
- **Interactive Web Interface:** A user-friendly front-end built with Streamlit allows for easy image uploads and provides clear, actionable results.
- **Advanced Analysis Tools:** Users can adjust the detection threshold to fine-tune the model's sensitivity for different types of imagery.
- **Detailed Quantitative Results:** The application provides key metrics, including the number of detected spill pixels, the percentage of the image covered by the spill, and an assessment of the spill's severity.
- **Visual Feedback:** The output includes a 4-quadrant plot showing the original image, a probability heatmap, the final binary mask, and a semi-transparent overlay for easy interpretation.

---

## 🛠️ Technology Stack

- **Language:** Python
- **Deep Learning Framework:** PyTorch
- **Core Libraries:**
  - `segmentation-models-pytorch`: For building the U-Net model with a pre-trained encoder.
  - `albumentations`: For high-performance data augmentation.
  - `OpenCV` & `Pillow`: For image processing and manipulation.
  - `scikit-image` & `matplotlib`: For analysis and plotting.
- **Web Framework:** Streamlit
- **Development Environment:** Google Colab, VS Code
- **Deployment:** GitHub (for code and model hosting via Releases), Streamlit Community Cloud

---

## 밟 Project Methodology & Milestones

The project was developed in a phased approach, with each model building upon the lessons learned from the previous one.

### Milestone 1: Data Collection and Preparation

- **Dataset:** The project utilizes a satellite image dataset where each image has a corresponding color-coded mask.
- **Mask Interpretation:** The model was trained to understand the specific color codes:
  - **Pink:** Oil Spill
  - **Cyan:** Water
  - **Yellow:** Land/Other Structures
  - **Black:** Background

### Milestone 2: Model Evolution and Training

We developed and trained three distinct models throughout the project's lifecycle:

1.  **Model 1 (Baseline Binary):** A simple binary segmentation model (Spill vs. No-Spill). While functional, it suffered from many false positives, often confusing dark water or land with oil.

2.  **Model 2 ("Improved" Binary):** This model used the same binary approach but with enhanced training techniques like aggressive data augmentation, a weighted loss function, and early stopping. It was more robust but still fundamentally limited by the binary classification approach.

3.  **Model 3 (Final Multi-Class):** This is the final, deployed model. It was trained to recognize all four classes from the masks.
    - **Architecture:** **U-Net** with a pre-trained **ResNet34** encoder (Transfer Learning).
    - **Loss Function:** **Cross-Entropy Loss**, the standard for multi-class problems.
    - **Result:** This model is far more accurate and intelligent, as it has learned the specific features of water and land, allowing it to distinguish them from oil spills reliably.

### Milestone 3: Evaluation

- The model's performance was validated both quantitatively (calculating metrics like Dice and IoU on a test set) and qualitatively (visualizing the output masks and overlays).

### Milestone 4: Deployment

- **Web Application:** A user interface was built using Streamlit, focusing on interactivity and clear presentation of results.
- **Model Hosting:** The large (94MB) trained model file (`multiclass_checkpoint.pth`) was hosted using **GitHub Releases** to overcome GitHub's file size limits.
- **Cloud Deployment:** The final application, including the `app.py` script and `requirements.txt` file, was deployed on **Streamlit Community Cloud** for permanent, public access.

---

## ⚙️ How to Run the Project Locally

### Prerequisites

- Git
- Python 3.9+
- `venv` (recommended)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yashita13/AI_SpillGuard_OSD-YashitaBahrani.git
    cd AI_SpillGuard_OSD-YashitaBahrani
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\Activate.ps1

    # For macOS/Linux
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will automatically download the trained model file on its first run and open in your web browser.

---

## 🚧 Challenges and Limitations

- **Dataset Limitation:** The model's performance is highly dependent on the training data. For production use, a much larger and more diverse dataset covering different weather conditions, geographic locations, and spill types would be required.
- **Distinguishing Look-Alikes:** The model may still struggle with rare natural phenomena that mimic oil spills, such as dense algae blooms.
- **Prototype System:** This application is a powerful proof-of-concept. A full production system would require a more complex backend for automated image ingestion and processing.
