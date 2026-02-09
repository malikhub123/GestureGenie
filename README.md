# ✨ GestureGenie – Real-Time Hand Gesture to Text and Speech Converter

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0-black?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A **computer vision and deep learning** project that recognizes **static hand gestures** of the English alphabet (A–Z) from a webcam feed and converts them into **text and speech** in real-time.

---


## 🌟 Overview

GestureGenie uses a custom-trained **Convolutional Neural Network (CNN)** to detect hand gestures, predict the corresponding letter, and enable users to build words and hear them spoken aloud through an interactive web interface.

Built completely **from scratch** – from data collection and preprocessing to model training and real-time deployment using Flask.

---

## 🚀 Demo

https://github.com/user-attachments/assets/459b6252-e8b8-4959-8bc7-1dc1b801a0f6

---

## ✨ Key Features

- 🎯 **Real-Time Gesture Recognition** – Detects hand gestures from webcam feed
- 🔤 **A-Z Alphabet Support** – Recognizes all 26 English alphabet letters
- 🗣️ **Text-to-Speech** – Converts built words to speech using gTTS
- 🖥️ **Interactive Web Interface** – User-friendly Flask-based UI
- 🤖 **Custom CNN Model** – Trained from scratch on self-collected dataset
- 📸 **Live Video Streaming** – Real-time prediction overlay on webcam feed
- ⌨️ **Word Building** – Add letters, spaces, delete, and speak words

---




### How It Works:

1. **Show a hand gesture** to the webcam
2. **GestureGenie detects** the hand using MediaPipe
3. **CNN predicts** the corresponding letter
4. **Build words** using the predicted letters
5. **Convert to speech** and hear your word!

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Deep Learning** | TensorFlow/Keras | Custom CNN model training and inference |
| **Computer Vision** | OpenCV | Image capture and preprocessing |
| **Hand Detection** | MediaPipe (cvzone) | Real-time hand tracking and detection |
| **Web Framework** | Flask | Backend server and live video streaming |
| **Text-to-Speech** | gTTS | Convert text to audio |
| **Audio Playback** | pygame | Play generated speech audio |
| **Frontend** | HTML, CSS, JavaScript | Interactive user interface |
| **Data Processing** | NumPy | Array operations and image manipulation |

---

## 📊 Project Pipeline

### 1️⃣ Data Collection
- Custom Python script using **OpenCV** and **MediaPipe** (via cvzone HandDetector)
- Captured hand images from webcam with bounding box detection
- Cropped and centered hand on white square background
- Organized into separate folders for each letter:
  ```
  Data/A, Data/B, Data/C, ..., Data/Z
  ```
- **900-1000 images** collected per letter

### 2️⃣ Data Preprocessing
Each captured hand image was:
- ✂️ Cropped tightly around the hand
- 📏 Resized proportionally
- 🎨 Placed on a fixed white canvas
- 🔄 Resized to **128×128** during training
- 🔢 Normalized to pixel range **[0, 1]**

### 3️⃣ Model Training
- Custom **CNN architecture** built with TensorFlow/Keras
- Dataset split:
  - **80% Training**
  - **20% Validation**
- **26 output classes** (A–Z)
- Training with data augmentation (rotation, zoom, flip)

### 4️⃣ Real-Time Prediction
- Same preprocessing pipeline applied to live webcam frames
- Processed image passed to trained model (`gesture_model.keras`)
- Model outputs probabilities for all 26 letters
- Highest probability letter displayed on video feed

### 5️⃣ Word Building & Speech
- **Flask web app** streams live webcam feed
- Interactive controls:
  - ➕ Add predicted letter to word
  - ⎵ Insert space
  - ⌫ Delete last letter
  - 🔊 Convert word to speech

---

## 🧠 Model Architecture

```
Input: 128×128×3 RGB Image
    ↓
Data Augmentation (Flip, Rotation, Zoom)
    ↓
Rescaling (1/255)
    ↓
Conv2D (32 filters, 3×3) + ReLU + MaxPooling
    ↓
Conv2D (64 filters, 3×3) + ReLU + MaxPooling
    ↓
Conv2D (128 filters, 3×3) + ReLU + MaxPooling
    ↓
GlobalAveragePooling2D
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (26 units, Softmax) → Output [A-Z]
```

**Loss Function:** `sparse_categorical_crossentropy`  
**Optimizer:** `Adam`  
**Metric:** `Accuracy`  

**Trained model saved as:** `gesture_model.keras`

---

## 📁 Project Structure

```
Gesture-Genie/
│
├── DataCollection.py        # Script to collect hand gesture images
├── app.py                   # Flask-based real-time prediction app
├── keras_model.h5           # Trained model (generated externally)
├── labels.txt               # Class labels (A–Z)
├── requirements.txt         # Project dependencies
├── templates/
│   └── index.html           # Frontend UI
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.8+** installed on your system
- Webcam for real-time gesture detection

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Gesture-Genie.git
cd Gesture-Genie
```

### 2️⃣ Install Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

**Dependencies include:**
```txt
tensorflow==2.12.1
opencv-python
mediapipe
cvzone
flask
gTTS
pygame
numpy
```

### 3️⃣ (Optional) Create Your Own Dataset

If you want to collect your own gesture images:

```bash
python DataCollection.py
```

**Instructions:**
- A webcam window will open
- Show the hand gesture for the desired letter
- Press `S` to save images
- Change the folder name in the script for each letter (e.g., `Data/A`, `Data/B`, ..., `Data/Z`)
- Collect 900-1000 images per letter for best results

⚠️ **Note:** This step is optional if you already have a trained model.

### 4️⃣ Train the Model (Optional)

If you collected your own dataset:

```bash
python train_model.py
```

This will:
- Load images from the `Data/` directory
- Train the CNN model
- Save the trained model as `gesture_model.keras`

### 5️⃣ Run the Application (if not collected your own dataset)

Start the Flask server:

```bash
python app.py
```

### 6️⃣ Open the Web Interface

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 🎮 How to Use

1. **Allow camera access** when prompted
2. **Show hand gestures** (A-Z) to the webcam
3. **See real-time predictions** on the video feed
4. **Click "Add Letter"** to add the predicted letter to your word
5. **Use controls:**
   - **Space** – Add a space between words
   - **Delete** – Remove the last character
   - **Speak** – Hear your word/sentence pronounced
6. **Build sentences** and have fun! 🎉

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Training Accuracy** | ~95%+ |
| **Validation Accuracy** | ~92%+ |
| **Real-Time FPS** | 15-25 FPS |
| **Inference Time** | ~50-80ms per frame |

> Results may vary based on lighting conditions and hand positioning

---


## 🚧 Challenges & Solutions

### Challenge 1: Inconsistent Hand Detection
**Solution:** Used MediaPipe with cvzone HandDetector for robust hand tracking and bounding box extraction.

### Challenge 2: Background Noise
**Solution:** Preprocessed images with consistent white background and proper cropping to reduce background interference.

### Challenge 3: Model Overfitting
**Solution:** Implemented data augmentation (rotation, zoom, flip) and dropout layers to improve generalization.

### Challenge 4: Real-Time Performance
**Solution:** Optimized image preprocessing pipeline and used GlobalAveragePooling instead of Flatten to reduce parameters.

---

## 🔮 Future Enhancements

- [ ] 🎯 Add support for **dynamic gestures** (continuous sign language)
- [ ] 🌍 Support for **multiple languages**
- [ ] 📱 **Mobile application** (Android/iOS)
- [ ] 🎨 Improved **UI/UX design**
- [ ] 📊 **Real-time accuracy metrics** display
- [ ] 🔄 **Model retraining** feature from the interface
- [ ] 🎥 **Video-to-text** conversion for pre-recorded videos
- [ ] 🧠 **Advanced models** (LSTM for sentence prediction)
- [ ] ☁️ **Cloud deployment** (Heroku, AWS, or Google Cloud)

---

## 📚 Learning Outcomes

This project demonstrates:

✅ End-to-end **deep learning pipeline**  
✅ **Computer vision** with OpenCV and MediaPipe  
✅ Custom **CNN architecture** design  
✅ **Data collection and preprocessing** techniques  
✅ **Real-time inference** and deployment  
✅ **Flask web development**  
✅ Integration of **third-party APIs** (gTTS)  
✅ **Model evaluation and optimization**  

---


## 👩‍💻 Author

**Aditi Malik**  
*B.Tech – Computer Science & Engineering*  
*Full-Stack Developer | AI/ML Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/aditi-malik-43880a222/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:malik2002.aditi@gmail.com)

---

## 📞 Contact

For questions, suggestions, or collaboration:
- 📧 Email: malik2002.aditi@gmail.com
- 💼 LinkedIn: [Your Profile](https://linkedin.com/in/aditi-malik-43880a222/)
- 🐙 GitHub: [@your-username](https://github.com/malikhub123)

---

## ⭐ Show Your Support

If you found this project helpful or interesting, please consider giving it a star! ⭐

---

<div align="center">

**Built with ❤️ and lots of hand gestures 👋**

Made by Aditi Malik | 2025

</div>
