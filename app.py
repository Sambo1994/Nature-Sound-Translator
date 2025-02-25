from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    if spectral_centroid < 2000:
        description = "The sound feels calm, like rustling leaves."
    elif spectral_centroid < 4000:
        description = "The sound carries energy, like birds chirping."
    else:
        description = "The sound is sharp, like a distant call."
    
    return {
        "description": description,
        "spectral_centroid": spectral_centroid,
        "zero_crossing_rate": zero_crossing_rate,
        "tempo": tempo
    }

@app.route("/translate", methods=["POST"])
def translate():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    audio_file = request.files["audio"]
    if not allowed_file(audio_file.filename):
        return jsonify({"error": "Invalid file format"}), 400
    
    if len(audio_file.read()) > MAX_FILE_SIZE:
        return jsonify({"error": "File size exceeds limit"}), 400
    audio_file.seek(0)  # Reset file pointer
    
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)
    
    analysis_result = analyze_audio(file_path)
    os.remove(file_path)  # Cleanup uploaded file
    
    return jsonify(analysis_result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
