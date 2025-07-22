# 🧠 Edge AI Prototype: Recyclable Item Classifier

This project demonstrates an **Edge AI** implementation for classifying recyclable items using a lightweight image classification model. The model is trained using TensorFlow, converted to **TensorFlow Lite**, and tested on a sample dataset to simulate deployment on an edge device like a Raspberry Pi.

---

## 📌 Objectives

- Train a small-scale image classification model suitable for edge devices  
- Convert the trained model to TensorFlow Lite format  
- Evaluate the model using accuracy metrics  
- Simulate edge deployment using Colab 
- Highlight Edge AI benefits for real-time applications

---

## 🛠 Tools and Technologies

- Python  
- TensorFlow & TensorFlow Lite  
- Google Colab / Raspberry Pi  
- NumPy, Matplotlib, PIL  

---

## 📂 Dataset

- Images are categorized into two classes: `Recyclable` and `Non-Recyclable`
- Each image is resized to `128x128`, normalized, and split into training and validation sets

---

## 🧪 Model Training Summary

- **Model Type:** Convolutional Neural Network (CNN)  
- **Input Shape:** 128x128x3  
- **Optimizer:** Adam  
- **Loss Function:** Binary or Categorical Crossentropy  
- **Epochs:** 10

### ✅ Model Accuracy
The metrcis for this are in the pdf file

## 🔄 Model Conversion to TensorFlow Lite

```python
import tensorflow as tf

# Load model and convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('model_path')
tflite_model = converter.convert()

# Save to file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

```

## ⚙️ Deployment Simulation

Simulated using **Google Colab** or **Raspberry Pi**:

1. Load `model.tflite`
2. Preprocess input image
3. Use TFLite interpreter for inference
4. Display predicted class

---

## 🚀 Why Edge AI?

Edge AI enables machine learning inference directly on local devices.

### ✅ Benefits:

- Real-time processing with low latency  
- No internet dependency (offline inference)  
- Better data privacy (no cloud upload)  
- Reduced bandwidth and power consumption

### 💡 Example Use Cases:

- Smart bins sorting

