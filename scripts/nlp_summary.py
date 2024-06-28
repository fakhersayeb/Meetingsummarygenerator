from pydub import AudioSegment
from pydub.silence import split_on_silence
import spacy
import sys
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
import speech_recognition as sr

# Load Spacy model once
nlp = spacy.load("en_core_web_sm")

# Initialize speech recognizer
recognizer = sr.Recognizer()

def recognize_speech_google(chunk):
    with sr.AudioFile(chunk) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service unavailable"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def split_audio(file_path):
    sound = AudioSegment.from_file(file_path, format="wav")
    chunks = split_on_silence(sound, min_silence_len=1000, silence_thresh=sound.dBFS-14, keep_silence=500)
    return chunks

def preprocess_text(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def textrank(sentences, num_sentences=2):
    # Preprocess text and remove stopwords
    stopwords = list(STOP_WORDS)
    clean_sentences = []
    for sent in sentences:
        words = [token.text for token in nlp(sent.lower()) if token.is_alpha and token.text not in stopwords]
        clean_sentences.append(" ".join(words))

    # Simplified PageRank algorithm
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity_matrix[i][j] = nlp(clean_sentences[i]).similarity(nlp(clean_sentences[j]))
            similarity_matrix[j][i] = similarity_matrix[i][j]

    scores = np.sum(similarity_matrix, axis=1)

    # Get top sentences based on scores
    top_sentence_indices = np.argsort(scores)[-num_sentences:]
    top_sentence_indices = sorted(top_sentence_indices)

    summary = " ".join([sentences[i] for i in top_sentence_indices])
    return summary

if __name__ == "__main__":
    audio_file = sys.argv[1]
    audio_chunks = split_audio(audio_file)

    transcript = ""
    for i, chunk in enumerate(audio_chunks):
        chunk_path = f"chunk{i}.wav"
        chunk.export(chunk_path, format="wav")
        transcript_chunk = recognize_speech_google(chunk_path)
        transcript += transcript_chunk + " "

    sentences = preprocess_text(transcript)
    summary = textrank(sentences)
    print(summary)
