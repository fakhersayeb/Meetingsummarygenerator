from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import whisper
import spacy
import sys

# Load Spacy model once
nlp = spacy.load("en_core_web_sm")

# Load tokenizer and a smaller model (distilbart-cnn-12-6)
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Load Whisper model
whisper_model = whisper.load_model("base")

def recognize_speech_whisper(file_path):
    result = whisper_model.transcribe(file_path)
    return result['text']

def split_audio(file_path):
    sound = AudioSegment.from_file(file_path, format="wav")
    chunks = split_on_silence(sound, min_silence_len=1000, silence_thresh=sound.dBFS-14, keep_silence=500)
    return chunks

def generate_summary(text):
    # Tokenize the text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Summarize the text using the smaller model
    summary = summarizer(" ".join(sentences), max_length=60, min_length=30, do_sample=False)[0]['summary_text']
    
    return summary

if __name__ == "__main__":
    audio_file = sys.argv[1]
    audio_chunks = split_audio(audio_file)
    
    transcript = ""
    for i, chunk in enumerate(audio_chunks):
        chunk_path = f"chunk{i}.wav"
        chunk.export(chunk_path, format="wav")
        transcript_chunk = recognize_speech_whisper(chunk_path)
        transcript += transcript_chunk + " "
    
    summary = generate_summary(transcript)
    print(summary)
