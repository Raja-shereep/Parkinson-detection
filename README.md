# 🎙️ Vocal Biomarker Screening Platform for Parkinson's Detection

This project is a **voice-based health screening web application** designed to detect early signs of **Parkinson’s Disease** using **machine learning** and **acoustic signal processing**.

---

## 🚀 Features

- 🎤 **Voice Upload or Live Recording** via browser
- 🎛️ **Feature Extraction** from voice: MFCCs, Chroma, ZCR, RMSE
- 🤖 **ML Model (Random Forest)** to predict presence of Parkinson’s
- 📈 **Dynamic Charts** for acoustic visualization
- ✅ **Confidence Score** with meaningful labels (High, Medium, Safe)
- 🌐 **Web UI** built with Flask + Plotly

---

## 🧠 Tech Stack

| Area              | Tools Used                          |
|-------------------|-------------------------------------|
| Frontend          | HTML, CSS, JavaScript, Plotly       |
| Backend           | Flask (Python)                      |
| ML & Signal Proc. | Scikit-learn, Librosa, SoundFile    |
| Visualization     | Plotly Graphs                       |
| Model Type        | Random Forest Classifier            |

---

## 📁 File Structure

├── app.py # Main Flask application
├── templates/
│ └── index.html # User Interface
├── model/
│ ├── model.pkl #Trained Random Forest model
│ └── scaler.pkl # StandardScaler for preprocessing
├── uploads/ # Temporarily stores user audio files
├── requirements.txt # Required packages
├── README.md # This file


## 📊 How It Works

1. **User records or uploads** a `.wav` audio sample.
2. Voice is preprocessed and **features are extracted**:
   - MFCCs (pitch & tone)
   - Chroma (frequency energy)
   - ZCR (voice tremors)
   - RMSE (energy/volume)
3. A **trained Random Forest model** analyzes features.
4. A **prediction & confidence score** are returned.
5. Results and **interactive charts** are shown to the user.


## ⚙️ Installation & Run

git clone https://github.com/yourusername/parkinsons-voice-detector.git
cd parkinsons-voice-detector
pip install -r requirements.txt
python app.py

Then visit: http://localhost:5000

📌 Dataset
Source: UCI Parkinson’s Dataset

195 voice samples with 22 acoustic features + target label

🔮 Future Enhancements
Support for more diseases (e.g., Alzheimer’s, Hypothyroidism)

Deep Learning model (CNN/LSTM) for improved accuracy

Streamlit or mobile-friendly deployment

Integration with telehealth APIs and wearable devices

📄 License
This project is for academic and educational purposes. Open-source contributions welcome.

