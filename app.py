from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
import sqlite3
import uuid
import gc  # For garbage collection

app = Flask(name)

# Ensure uploads folder exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create / connect to database
conn = sqlite3.connect('emotions.db', check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        image_path TEXT,
        emotion TEXT
    )
""")
conn.commit()

# Hardcoded emotion labels
labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        image_file = request.files['image']

        # --- Clear old uploads ---
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # --- Save new image with unique filename ---
        unique_filename = f"{uuid.uuid4().hex}_{image_file.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        image_file.save(image_path)

        # --- Read and preprocess image ---
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # --- Load model lazily to reduce memory usage ---
        model = tf.keras.models.load_model("emotion_model.h5")

        # --- Predict emotion ---
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        emotion = labels[class_index]

        # --- Save to database ---
        cursor.execute(
            "INSERT INTO users (name, image_path, emotion) VALUES (?, ?, ?)",
            (name, image_path, emotion)
        )
        conn.commit()

        # --- Clean up model from memory ---
        del model
        gc.collect()

        # --- Render result ---
        return render_template('index.html', result=emotion, img=image_path)

    except Exception as e:
        # Show error instead of crashing
        return f"An error occurred: {str(e)}"

if name == 'main':
    app.run(debug=True)
