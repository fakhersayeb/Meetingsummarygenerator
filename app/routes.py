from flask import request, jsonify, render_template, current_app as app
import os
import speech_recognition as sr
from werkzeug.utils import secure_filename
import spacy
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

ALLOWED_EXTENSIONS = {'wav', 'aiff', 'aif'}

# Load Spacy model once
nlp = spacy.load("en_core_web_sm")

# Initialize speech recognizer
recognizer = sr.Recognizer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recognize_speech_google(file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service unavailable"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def preprocess_text(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def textrank(sentences, num_sentences=2):
    stopwords = list(STOP_WORDS)
    clean_sentences = []
    for sent in sentences:
        words = [token.text for token in nlp(sent.lower()) if token.is_alpha and token.text not in stopwords]
        clean_sentences.append(" ".join(words))

    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity_matrix[i][j] = nlp(clean_sentences[i]).similarity(nlp(clean_sentences[j]))
            similarity_matrix[j][i] = similarity_matrix[i][j]

    scores = np.sum(similarity_matrix, axis=1)
    top_sentence_indices = np.argsort(scores)[-num_sentences:]
    top_sentence_indices = sorted(top_sentence_indices)
    summary = " ".join([sentences[i] for i in top_sentence_indices])
    return summary

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

        transcript = recognize_speech_google(file_path)
        sentences = preprocess_text(transcript)
        summary = textrank(sentences)

        return jsonify({'summary': summary})
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == "__main__":
    app.run(debug=True)
