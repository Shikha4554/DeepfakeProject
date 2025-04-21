from flask import Flask, render_template, request, redirect, session
import os
import random
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load fine-tuned model
print("⏳ Loading fine-tuned model...")
processor = AutoImageProcessor.from_pretrained("finetuned-deepfake-detector")
model = AutoModelForImageClassification.from_pretrained("finetuned-deepfake-detector")
print("✅ Model loaded successfully.")

# Predict DeepFake
def predict_deepfake(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        class_id = torch.argmax(probs).item()
        confidence = round(probs[0][class_id].item() * 100, 2)

    label = model.config.id2label[class_id].capitalize()
    return label, confidence

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(path)

            label, confidence = predict_deepfake(path)
            session['image_path'] = uploaded_file.filename
            session['label'] = label
            session['confidence'] = confidence

            return redirect('/result')
    return render_template('upload.html')

@app.route('/result')
def result():
    image_path = session.get('image_path')
    label = session.get('label')
    confidence = session.get('confidence')

    return render_template('result.html', image_path=image_path, label=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
