# Fall Detection in Public Spaces
**By Eden Amram & Ziv Shamli**

Real-time fall detection system for outdoor and crowded environments using deep learning.


## 📌 Overview

Falls are a major cause of unintentional injury worldwide, posing significant health risks in both indoor and outdoor environments. While existing fall-detection models perform well in controlled indoor settings, their accuracy often drops in real-world public spaces due to variations in lighting, backgrounds, camera angles, and human appearances.

Collecting large-scale real-world fall data is challenging, which motivates the use of generative AI models to create synthetic datasets. This project evaluates several generative approaches, including **Text-to-Image**, **Image-to-Image**, and **Inpainting** techniques, to produce realistic fall scenarios in public spaces.

The generated datasets were used to fine-tune **LBFD-Net**, a lightweight CNN optimized for real-time fall detection on edge devices. Results show that **inpainting-based synthetic data**, which preserves scene structure and reduces generative artifacts, provides the best performance across **accuracy**, **precision**, **F1-score**, **AUC** and **ROC** metrics. Fully generative models like **Stable Diffusion** and **Nano Banana**, while visually realistic, exhibited lower generalization due to domain shift and structural inconsistencies.

These findings demonstrate that task-aware and physically plausible synthetic data can significantly enhance fall detection models, particularly when real-world datasets are limited or unavailable. The study emphasizes the importance of domain-aligned data generation for safety-critical computer vision applications and offers guidance for future research on synthetic data and alternative fall detection methods.

---


## 🎯 Problem Statement

Existing fall detection systems are limited in real-world applications because they are mostly trained on indoor datasets under controlled conditions. Outdoor and crowded environments introduce challenges such as varying lighting, complex backgrounds, occlusions, and diverse human appearances, which existing models are not robust to.

The main limitation is the lack of large-scale, annotated outdoor fall datasets. Collecting real-world fall data in public spaces is difficult due to safety, ethical, and privacy concerns. Consequently, most models struggle to generalize beyond indoor environments and fail to maintain high accuracy, sensitivity, and robustness in realistic scenarios.

**Research Question:**

> “How does training with generative images of outdoor and crowded environments impact the generalization, accuracy, sensitivity, and robustness of a fall detection model across indoor and outdoor settings?”

This project leverages synthetic datasets generated using **Text-to-Image**, **Image-to-Image**, and **Inpainting** methods to enhance model performance and generalization, enabling reliable fall detection across both indoor and outdoor environments.

---
## 🧠 Methodology

This research consists of **two main phases** aimed at improving fall detection in public spaces using synthetic data.



### Phase 1: Generative Model Evaluation and Synthetic Dataset Creation

Due to the **lack of publicly available outdoor or crowded fall datasets**, we first generated synthetic fall images using advanced generative AI models. The goal was to create realistic images that are free of artifacts and suitable for training the **LBFD-Net** model.

**Generative Model Evaluation** was conducted through three tracks:

#### Track A: Text-to-Image

* Each model generated 10 images based on textual descriptions of fall events in public spaces.
* **Physical Metrics (High Priority)**:
  * **Depth Consistency** – spatial coherence of scene depth and logical integration of body poses.
  * **Shading Consistency** – alignment of shadows with lighting.
  * **Surface Normals Consistency** – ensures objects and human figures are geometrically plausible.
* **Perceptual Metrics (Secondary Priority)**:
  * **NIQE, BRISQUE, MA Score** – evaluate naturalness, spatial quality, and aesthetic composition.

#### Track B: Image-to-Image

* Generated 10 images per model using reference fall/non-fall images.
* Focused on **physical plausibility** due to content variation in generation.
* Metrics prioritized **Depth, Shading, Normals**, followed by **NIQE**.

#### Track C: Inpainting-Based Images

* Humans were segmented using **YOLO + SAM**, then inserted into public-space backgrounds.
* Inpainting completed the surrounding areas ensuring **lighting, perspective, texture, and scene consistency**.
* Metrics: same hierarchy as above (**Depth, Shading, Normals**, then **NIQE**).



### Synthetic Dataset Generation

* Top-performing models from the three tracks were used to generate **datasets of 300–330 labeled images each**.
* Images depict **fall and non-fall scenarios** across diverse public environments.
* Key variations included:
  * Background complexity
  * Lighting conditions
  * Camera angles
  * Human poses and crowd presence
* All images were **manually labeled** for high accuracy.
* These datasets were used for **fine-tuning and testing LBFD-Net**, enabling evaluation of model generalization across scenarios.



### Phase 2: Fall Detection Using LBFD-Net

#### Datasets

* **Inpainting synthetic dataset** – realistic fall/non-fall figures embedded in public-space scenes.
* **Stable Diffusion dataset** – high variability in lighting and backgrounds.
* **Nano Banana dataset** – stylistically diverse synthetic images.
* **Real indoor dataset** – serves as a baseline and reference.

#### LBFD-Net Fine-Tuning

* Base model: **LBFD-Net pretrained on indoor data**.
* Fine-tuning conducted independently on each synthetic dataset and on a combined dataset:
  1. **LBFD-Net fine-tuned on Inpainting Dataset**
  2. **LBFD-Net fine-tuned on Stable Diffusion Dataset**
  3. **LBFD-Net fine-tuned on Nano Banana Dataset**
  4. **LBFD-Net fine-tuned on all synthetic datasets combined**
* **No weights frozen** in second residual block or classifier.
* **Training Settings**:
  * Framework: PyTorch
  * Loss: Binary Cross-Entropy with Logits
  * Optimizer: Adam, LR = 5×10⁻⁵
  * StepLR scheduler: decay every 4 epochs
  * Batch size: 16
* Each epoch: forward pass → loss → backpropagation → parameter update.

#### Validation and Model Selection

* Validation per epoch computes **loss, accuracy, precision, recall, and F1-score**.
* Final model selected based on **lowest validation loss** with early stopping to prevent overfitting.

#### Test Set Design

* Test sets include **real indoor images not seen in training** + **synthetic images** from the same generative model.
* Additional **general test set** combines all synthetic models and indoor images for comprehensive evaluation.

#### Evaluation Metrics

* **Classification metrics**: Accuracy, Precision, Recall, F1-score
* **Curve-based metrics**: ROC curve and AUC for threshold-independent performance assessment

---
## 📂 Datasets

This project utilizes both real-world and synthetic datasets to evaluate fall detection performance across indoor and outdoor environments.



### 🏠 Indoor Dataset

A real-world indoor dataset containing labeled fall and non-fall scenarios.
This dataset serves as the **baseline training data** and reference domain.

- Description: Controlled indoor environment with predefined fall actions.
- Usage: Pretraining of LBFD-Net and part of the test sets.
- Link: [Indoor Fall Dataset](https://drive.google.com/drive/folders/1SG5pySW-bwBZHMDotKXn0ELFxnFgY9Dd?usp=sharing)



### 🎨 Synthetic Datasets

Synthetic datasets were generated to address the lack of outdoor and crowded fall data.

Each generative model produced an independent labeled dataset (300–330 images per model).

#### 🖌 Inpainting-Based Dataset
- Human subjects segmented using YOLO + SAM and embedded into real public-space backgrounds.
- Preserves scene structure and physical consistency.
- Link: [Inpainting Synthetic Dataset](https://drive.google.com/drive/folders/1Vu9QyxuEfRC6UY32gaA3FKjVRNkz3ZBk?usp=sharing)

#### 🌌 Stable Diffusion Dataset
- Fully generative Text-to-Image approach.
- Includes diverse outdoor scenes (streets, parks, sidewalks).
- High visual variability.
- Link: [Stable Diffusion Dataset](https://drive.google.com/drive/folders/1UzhWluflQEMK-F1UyVaqg839xBpJRJ0u?usp=sharing)

#### 🍌 Nano Banana Dataset
- Fully generative dataset emphasizing stylistic diversity.
- Includes crowded public-space environments.
- Link: [Nano Banana Dataset](https://drive.google.com/drive/folders/1er4N-wguxG4WqTwwhVcd-BAtotWRJAYJ?usp=sharing)



#### 🧪 Combined Synthetic Dataset

A merged dataset containing all synthetic images from the three generative approaches.
Used for joint fine-tuning experiments.

- Link: [Combined Synthetic Dataset](https://drive.google.com/drive/folders/1exBLVg8oqrWU98lZD3bgMLcxzKFPHUWR?usp=sharing)

---

## ⚙️ Installation

This project consists of two main parts (Generative models Evaluation and Fall Detection Training).  
For both parts, it is recommended to create a dedicated **virtual environment** and install the required dependencies.


### Create a Virtual Environment

```bash
python -m venv venv
```

 ### Activate the environment:
 **Windows:**
 ```bash
venv\Scripts\activate
```
**Linux:**
```bash
source venv/bin/activate
```
### Install Pytorch with your cuda version 

PyTorch must match the CUDA version installed on your system.

**Check Your CUDA Version**

```bash
nvcc --version
```
**Install PyTorch**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu[YOUR_VERSION]
```
###  Generative Models Evaluation Installation
```bash
cd /Generative models evaluation
```
#### 📦 Install Required Dependencies
```bash
pip install -r requirements.txt
```
###  LBFD-NET Installation
```bash
cd /LBFD-Net
```
#### 📦 Install Required Dependencies
```bash
pip install -r requirements.txt
```
#### 📒 Install Jupyter Notebook 

```bash
pip install notebook
```

---

