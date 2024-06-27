from flask import request, jsonify, render_template, current_app as app
import os
import speech_recognition as sr
from werkzeug.utils import secure_filename
from transformers import pipeline
import spacy

ALLOWED_EXTENSIONS = {'wav', 'aiff', 'aif'}

# Load Spacy model once
nlp = spacy.load("en_core_web_sm")

# Load a smaller and faster summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Recognize speech using Google Web Speech API
        transcript = recognize_speech_google(file_path)

        # Generate summary
        summary = generate_summary(transcript)

        return jsonify({'summary': summary})

    else:
        return jsonify({'error': 'File type not allowed'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recognize_speech_google(file_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file_path)
    with audio_file as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service unavailable"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def generate_summary(text):
    # Tokenize the text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Summarize the text using the pre-trained transformer model
    summary = summarizer(" ".join(sentences), max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    
    return summary
