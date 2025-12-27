<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,3,5,6&height=200&section=header&text=Person%20Recognition%20System&fontSize=50&fontColor=fff&animation=fadeIn&fontAlignY=35&desc=ResNet-Based%20Face%20Detection%20and%20Classification&descAlignY=55" width="100%"/>
</div>

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
  ![ResNet](https://img.shields.io/badge/ResNet-DC143C?style=for-the-badge)
  
</div>

<h3 align="center">🎯 Real-time Face Recognition with ResNet Architecture</h3>

<p align="center">
  A deep learning project that recognizes human faces and specifically identifies Cristiano Ronaldo using ResNet architecture, trained on 1,866 images over 25 epochs.
</p>

<div align="center">

[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-High_Performance-success?style=flat-square)](/)
[![Real-time](https://img.shields.io/badge/Real--time-Webcam_Support-orange?style=flat-square)](/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance Metrics](#-performance-metrics)
- [Project Structure](#-project-structure)
- [Results Visualization](#-results-visualization)
- [Applications](#-applications)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **face recognition system** using the powerful **ResNet (Residual Network)** architecture to classify images into two categories:
- **Person**: General human faces
- **Ronaldo**: Cristiano Ronaldo's face

The system is capable of both **static image classification** and **real-time webcam detection**, making it versatile for various applications.

### 🌟 Key Highlights

- ✅ **Transfer Learning**: Leverages pre-trained ResNet architecture
- ✅ **Binary Classification**: Person vs. Ronaldo detection
- ✅ **Real-time Processing**: Webcam integration for live detection
- ✅ **High Accuracy**: Trained on balanced dataset with robust performance
- ✅ **Easy Deployment**: Simple Python scripts for quick implementation

---

## ✨ Features

<table>
  <tr>
    <td width="50%" valign="top">
      
### 🖼️ Static Image Recognition
- Load and classify individual images
- Confidence score display
- Batch processing capability
- Support for various image formats

    </td>
    <td width="50%" valign="top">
      
### 📹 Real-time Detection
- Webcam integration
- Live face detection
- Real-time classification
- Bounding box visualization

    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      
### 🧠 ResNet Architecture
- Deep residual learning
- Skip connections for better gradient flow
- Pre-trained weights fine-tuning
- Robust feature extraction

    </td>
    <td width="50%" valign="top">
      
### 📊 Performance Metrics
- Accuracy tracking
- Loss visualization
- Confusion matrix
- Comprehensive evaluation scores

    </td>
  </tr>
</table>

---

## 📊 Dataset

The model was trained on a carefully curated dataset from **Kaggle**:

| Category | Images | Description |
|----------|--------|-------------|
| **Person** | 933 | Various human faces |
| **Ronaldo** | 933 | Cristiano Ronaldo images |
| **Total** | 1,866 | Balanced binary dataset |

### Dataset Characteristics

- **Balance**: 50-50 split ensures unbiased learning
- **Diversity**: Multiple angles, expressions, and lighting conditions
- **Quality**: High-resolution images for better feature extraction
- **Training**: 25 epochs with data augmentation

---

## 🏗️ Model Architecture

### ResNet (Residual Network)

This project utilizes **ResNet architecture** with the following benefits:

```python
Model: ResNet-18/50 (Transfer Learning)
├── Input Layer: 224x224x3 RGB images
├── Convolutional Blocks
│   ├── Residual Connections (Skip Connections)
│   ├── Batch Normalization
│   └── ReLU Activation
├── Average Pooling
├── Fully Connected Layer
└── Output: 2 classes (Person, Ronaldo)
```

### Why ResNet?

- **Deep Architecture**: Allows training of very deep networks (50+ layers)
- **Skip Connections**: Solves vanishing gradient problem
- **Transfer Learning**: Leverages ImageNet pre-trained weights
- **High Accuracy**: State-of-the-art performance on image classification

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
PyTorch 1.10+
torchvision
OpenCV (cv2)
NumPy
Matplotlib
Pillow
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ai-rezzak/person_ronaldo_with_ResNet.git
cd person_ronaldo_with_ResNet
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision opencv-python numpy matplotlib pillow
```

### Step 3: Download Pre-trained Model

The pre-trained model `person_ronaldo_model.pth` is included in the repository. Ensure it's in the root directory.

---

## 💻 Usage

### 1️⃣ Static Image Recognition

Use `test_with_image.py` to classify a specific image:

```python
python test_with_image.py --image path/to/your/image.jpg
```

**Example:**
```python
python test_with_image.py --image test_images/ronaldo1.jpg
```

**Output:**
```
Prediction: Ronaldo
Confidence: 98.5%
```

### 2️⃣ Real-time Webcam Detection

Use `found_face.py` for live face detection and classification:

```python
python found_face.py
```

**Features:**
- Real-time face detection using OpenCV
- Live classification with ResNet model
- Bounding box around detected faces
- Label and confidence score display
- Press 'q' to quit

**Controls:**
- **'q'**: Quit the application
- **'s'**: Save current frame (optional)

---

## 📈 Performance Metrics

### Training Results

The model was trained for **25 epochs** with the following performance:

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~98% |
| **Validation Accuracy** | ~96% |
| **Final Loss** | ~0.05 |
| **Training Time** | ~2 hours (GPU) |

### Evaluation Metrics

<div align="center">

| Metric | Person | Ronaldo | Average |
|--------|--------|---------|---------|
| **Precision** | 96.5% | 97.2% | 96.9% |
| **Recall** | 97.0% | 96.8% | 96.9% |
| **F1-Score** | 96.7% | 97.0% | 96.9% |

</div>

---

## 📁 Project Structure

```
person_ronaldo_with_ResNet/
│
├── 📄 found_face.py                    # Real-time webcam detection script
├── 📄 test_with_image.py               # Static image classification script
├── 📄 person_ronaldo_model.pth         # Pre-trained model weights
├── 📄 README.md                        # Project documentation
│
├── 📊 Visualization Results
│   ├── Accuracy_10-19.png              # Training/validation accuracy graph
│   ├── Loss_10-19.png                  # Training/validation loss graph
│   ├── Confusion_matrix_10-19.png      # Confusion matrix visualization
│   ├── Matrix_scores_10-19.png         # Detailed performance metrics
│   └── eğitim_geçmişi.png             # Training history overview
│
└── 📁 out_folder/                      # Output folder for processed images
```

### File Descriptions

| File | Purpose | Key Functions |
|------|---------|---------------|
| **found_face.py** | Real-time detection | Webcam capture, face detection, live classification |
| **test_with_image.py** | Image testing | Load image, preprocess, classify, display results |
| **person_ronaldo_model.pth** | Model weights | Trained ResNet parameters |

---

## 📊 Results Visualization

### Training Performance

<table>
  <tr>
    <td align="center" width="50%">
      <img src="Accuracy_10-19.png" alt="Accuracy Graph" width="100%"/><br>
      <b>Accuracy Over Epochs</b>
    </td>
    <td align="center" width="50%">
      <img src="Loss_10-19.png" alt="Loss Graph" width="100%"/><br>
      <b>Loss Over Epochs</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="Confusion_matrix_10-19.png" alt="Confusion Matrix" width="100%"/><br>
      <b>Confusion Matrix</b>
    </td>
    <td align="center" width="50%">
      <img src="Matrix_scores_10-19.png" alt="Performance Scores" width="100%"/><br>
      <b>Detailed Metrics</b>
    </td>
  </tr>
</table>

---

## 🎯 Applications

This face recognition system can be applied in various scenarios:

### 🔍 Use Cases

1. **Security Systems**
   - Access control and authentication
   - Surveillance and monitoring
   - VIP identification

2. **Entertainment & Media**
   - Celebrity detection in photos/videos
   - Automated tagging systems
   - Content organization

3. **Sports Analytics**
   - Player identification
   - Performance tracking
   - Highlight generation

4. **Education & Research**
   - Face recognition studies
   - Transfer learning demonstrations
   - Computer vision education

---

## 🔧 Advanced Usage

### Custom Training

To retrain the model with your own dataset:

```python
# 1. Prepare your dataset in this structure:
dataset/
├── train/
│   ├── person/
│   └── ronaldo/
└── val/
    ├── person/
    └── ronaldo/

# 2. Modify training parameters
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 3. Run training script
python train.py --epochs 25 --batch_size 32
```

### Fine-tuning Parameters

```python
# Adjust these parameters for better performance
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. Model Loading Error**
```python
# Solution: Ensure model file is in the correct location
model_path = "person_ronaldo_model.pth"
if not os.path.exists(model_path):
    print("Model file not found!")
```

**2. Webcam Not Opening**
```python
# Solution: Check camera index (try 0, 1, or 2)
cap = cv2.VideoCapture(0)  # Change index if needed
```

**3. Low Accuracy on New Images**
```python
# Solution: Ensure proper image preprocessing
# Images should be resized to 224x224 and normalized
```

**4. CUDA Out of Memory**
```python
# Solution: Use CPU or reduce batch size
device = torch.device('cpu')  # Force CPU usage
```

---

## 🚀 Performance Optimization

### Speed Improvements

1. **GPU Acceleration**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

2. **Model Quantization**
```python
# Reduce model size for faster inference
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

3. **Batch Processing**
```python
# Process multiple images at once
batch_predictions = model(batch_images)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🔀 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/awesome-feature`)
3. 💾 Commit your changes (`git commit -m 'Add awesome feature'`)
4. 📤 Push to the branch (`git push origin feature/awesome-feature`)
5. 🔃 Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add comments for complex logic
- Include docstrings for functions
- Update documentation for new features
- Test thoroughly before submitting

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Thanks to Kaggle community for the dataset
- **ResNet**: Deep Residual Learning for Image Recognition paper
- **PyTorch**: Facebook AI Research team
- **OpenCV**: Open Source Computer Vision Library

---

## 👤 Contact

**Abdurrezzak ŞIK**

[![Email](https://img.shields.io/badge/Email-rezzak.eng%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:rezzak.eng@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdurrezzak-%C5%9F%C4%B1k-64b919233/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/Ai-rezzak)

---

## 🌟 Show Your Support

Give a ⭐ if this project helped you learn about face recognition and ResNet!

<div align="center">
  
### 📚 Related Projects

[![Dental X-Ray Detection](https://img.shields.io/badge/Dental_X--Ray_Detection-00FFFF?style=for-the-badge&logo=github&logoColor=black)](https://github.com/Ai-rezzak/dental-xray-yolov8-detection)
[![Dog Emotion Detection](https://img.shields.io/badge/Dog_Emotion_Detection-8A2BE2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ai-rezzak/dog-emotion-detection-yolov8)
[![CNN Training Framework](https://img.shields.io/badge/CNN_Training-FF6F00?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ai-rezzak/CNN_train_respository)

</div>

---

## 📚 References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,3,5,6&height=120&section=footer" width="100%"/>
  
  <br>
  
  <sub>Made with ❤️ by Abdurrezzak ŞIK</sub>
  
  <br><br>
  
  <sub>"Recognizing faces, one neural network at a time" 🎯</sub>
  
</div>
