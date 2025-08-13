from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from util import load_artifacts, classify_waste
from complaints_app import complaints_app
from complaints_app.complaints_app import complaints_app

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Configuration for file upload and complaints data
app.config['UPLOAD_FOLDER'] = 'static/complaints_data'
app.config['COMPLAINTS_FOLDER'] = 'complaints_data'
app.static_folder = 'static'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 

# Ensure the folders exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['COMPLAINTS_FOLDER']):
    os.makedirs(app.config['COMPLAINTS_FOLDER'])

# Register the complaints blueprint with a URL prefix of /complaints
app.register_blueprint(complaints_app, url_prefix='/complaints')

# Load model artifacts when the app starts
load_artifacts()

# Define the allowed image file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Set up the folder for storing uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading an image and classifying it
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Classify the image using the loaded model
        predicted_class, description, video_embeds, image_id = classify_waste(filepath)

        # Construct the video URLs/Embeds (if more than one video is found)
        video_embed_html = ''.join(video_embeds)  # Embed all videos in HTML format

        # Return response with image URL, predictions, description, and video embeds
        response_data = {
            "image_url": filepath,
            "predicted_class": predicted_class,
            "description": description,
            "video_embeds": video_embed_html,  # Embed YouTube videos here
            "image_preview": f"{UPLOAD_FOLDER}/{filename}"
        }

        return jsonify(response_data), 200

    return jsonify({"error": "Invalid file format"}), 400

# Route for NGOs & Recycling Companies page
@app.route('/ngo-recycling')
def ngo_recycling():
    return render_template('ngo_recycling.html')

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
