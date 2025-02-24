from flask import Flask, request
import librosa
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    if spectral_centroid < 2000:
        return "The sound feels calm, like rustling leaves."
    elif spectral_centroid < 4000:
        return "The sound carries energy, like birds chirping."
    else:
        return "The sound is sharp, like a distant call."

@app.route("/translate", methods=["POST"])
def translate():
    if "audio" not in request.files:
        return "No file uploaded", 400
    
    audio_file = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    translation = analyze_audio(file_path)
    return translation

if __name__ == "__main__":
   port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
