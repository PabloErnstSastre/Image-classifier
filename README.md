# CIFAR-10 Image Classifier with CNN

Lightweight Convolutional Neural Network (CNN) built with PyTorch to classify images across the 10 classes of the CIFAR-10 dataset. The model was trained and evaluated locally using Python 3.13 (PyTorch CPU-only environment)[cite: 6].

---

## 1. Network Architecture
The model consists of a custom CNN structure featuring[cite: 6]:
* **Convolutional Blocks:** 3x `Conv2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool` $\rightarrow$ `Dropout`[cite: 6].
* **Fully-Connected Layers:** `Linear` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout` $\rightarrow$ Output layer[cite: 6].

---

## 2. Training Results
* **Epochs:** 5[cite: 6]
* **Best Validation Loss:** 0.7673[cite: 6]
* **Best Validation Accuracy:** 72.70%[cite: 6]

---

## 3. Test Set Evaluation
* **Overall Accuracy:** 73.24%[cite: 6]

### Classification Report (Metrics by Class)

| Class | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: |
| **airplane** | 0.76[cite: 6] | 0.68[cite: 6] | 0.72[cite: 6] |
| **automobile** | 0.84[cite: 6] | 0.90[cite: 6] | 0.87[cite: 6] |
| **bird** | 0.56[cite: 6] | 0.64[cite: 6] | 0.60[cite: 6] |
| **cat** | 0.63[cite: 6] | 0.51[cite: 6] | 0.56[cite: 6] |
| **deer** | 0.76[cite: 6] | 0.62[cite: 6] | 0.68[cite: 6] |
| **dog** | 0.68[cite: 6] | 0.66[cite: 6] | 0.67[cite: 6] |
| **frog** | 0.79[cite: 6] | 0.80[cite: 6] | 0.80[cite: 6] |
| **horse** | 0.77 | 0.78 | 0.77 |
| **ship** | 0.71 | 0.93 | 0.80[cite: 7] |
| **truck** | 0.84[cite: 7] | 0.80[cite: 7] | 0.82[cite: 7] |

---

## 4. Inference (`predict.py`)
You can use `predict.py` with any custom image to get the predicted class along with a probability distribution chart[cite: 7].

### Usage
```bash
python predict.py --image_path ./my_image.jpg
