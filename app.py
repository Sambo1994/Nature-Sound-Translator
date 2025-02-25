from flask import Flask, request, jsonify
from flask_cors import CORS  # Added CORS support
import librosa
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Enable CORS for specific frontend domain (replace with your frontend domain)
CORS(app, origins=["https://your-frontend-domain.com"])

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for audio files
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/translate', methods=['POST'])
def translate_sound():
    # Check if the file is present in the request
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['audio']
    
    # Check if a file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if the file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload a WAV, MP3, OGG, or FLAC file.'}), 400
    
    # Save the file securely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Process the audio file
        y, sr = librosa.load(filepath, sr=None)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Clean up the uploaded file after processing
        os.remove(filepath)
        
        # Return the results
        return jsonify({
            'description': 'Processed nature sound',
            'spectral_centroid': float(spectral_centroid),  # Convert to float for JSON serialization
            'zero_crossing_rate': float(zero_crossing_rate),
            'tempo': float(tempo)
        })
    except Exception as e:
        # Handle any errors during processing
        return jsonify({'error': f'Error processing audio file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
