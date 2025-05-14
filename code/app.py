from flask import Flask, request, render_template
import numpy as np
import librosa
import joblib
import os
import plotly.graph_objects as go

app = Flask(__name__)

# Load your trained model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Extract features (modified to avoid using mean for ZCR and RMSE)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr_full = librosa.feature.zero_crossing_rate(y)[0]
    rmse_full = librosa.feature.rms(y=y)[0]

    zcr = np.mean(zcr_full)
    rmse = np.mean(rmse_full)

    features = np.hstack([mfccs, chroma, zcr, rmse])

    return features.reshape(1, -1), zcr_full, rmse_full

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Extract features and ZCR/RMSE arrays
            features, zcr_arr, rmse_arr = extract_features(filepath)

            # Ensure that we only use the first 22 features
            if features.shape[1] != 22:
                if features.shape[1] > 22:
                    features = features[:, :22]  # Trimming to 22 features
                else:
                    return f"Error: Extracted {features.shape[1]} features, but model expects 22. Please check extraction."

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict
            prediction = model.predict(features_scaled)
            prediction_proba = model.predict_proba(features_scaled)

            confidence = np.max(prediction_proba) * 100

            if prediction[0] == 1 and confidence >= 60:
                result = f"❗High chance of Parkinson’s detected: {confidence:.2f}%"
            elif prediction[0] == 1:
                result = f"⚠️ Mild signs of Parkinson’s: {confidence:.2f}%"
            else:
                result = f"✅ No signs of Parkinson’s detected: {confidence:.2f}%"

            # Create the charts
            fig_mfcc = go.Figure(data=go.Scatter(x=np.arange(13), y=features[0, :13], mode='lines', name='MFCCs'))
            fig_chroma = go.Figure(data=go.Scatter(x=np.arange(13), y=features[0, 13:26], mode='lines', name='Chroma'))
            fig_zcr = go.Figure(data=go.Scatter(x=np.arange(len(zcr_arr)), y=zcr_arr, mode='lines', name='Zero Crossing Rate'))
            fig_rmse = go.Figure(data=go.Scatter(x=np.arange(len(rmse_arr)), y=rmse_arr, mode='lines', name='RMSE'))

            fig_mfcc.update_layout(title="MFCCs", plot_bgcolor="rgb(240, 240, 240)")
            fig_chroma.update_layout(title="Chroma", plot_bgcolor="rgb(240, 240, 240)")
            fig_zcr.update_layout(title="Zero Crossing Rate", plot_bgcolor="rgb(240, 240, 240)")
            fig_rmse.update_layout(title="RMSE", plot_bgcolor="rgb(240, 240, 240)")

            return render_template(
                'index.html',
                result=result,
                prediction_prob=confidence,
                fig_mfcc=fig_mfcc.to_html(full_html=False),
                fig_chroma=fig_chroma.to_html(full_html=False),
                fig_zcr=fig_zcr.to_html(full_html=False),
                fig_rmse=fig_rmse.to_html(full_html=False)
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
