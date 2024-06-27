import subprocess

def generate_summary(transcript_path):
    result = subprocess.run(['python3', 'scripts/nlp_summary.py', transcript_path], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')
