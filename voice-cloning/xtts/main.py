import os
import soundfile as sf
import uuid
import requests  # Для загрузки по HTTP
from google.cloud import storage
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
import nltk

# Ensure NLTK is downloaded
nltk.download('punkt')

# Initialize TTS model
def load_model():
    config = XttsConfig()
    config.load_json("./XTTS-v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="./XTTS-v2/", eval=True)
    model.cuda()
    return model, config

model, config = load_model()

# Initialize Google Cloud Storage client
storage_client = storage.Client()
BUCKET_NAME = "your-bucket-name"  # Replace with your bucket name

def download_reference_audio(reference_url):
    """Download reference audio from Google Cloud Storage or HTTP URL."""
    temp_file = f"/tmp/{uuid.uuid4()}.wav"
    
    if reference_url.startswith("https://storage.googleapis.com"):
        # Download the file using HTTP
        response = requests.get(reference_url, stream=True)
        response.raise_for_status()  # Raise an error if download fails

        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise ValueError("Unsupported URL format. Provide a valid GCS or HTTP link.")

    return temp_file

def upload_to_bucket(file_path):
    """Upload the audio file to Google Cloud Storage and return the public URL."""
    file_name = os.path.basename(file_path)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_path)
    blob.make_public()  # Make the file publicly accessible
    return blob.public_url

def process_tts_sentence(input_text, reference_audio, output_file):
    """Generate audio for a single sentence."""
    outputs = model.synthesize(
        text=input_text,
        config=config,
        speaker_wav=[reference_audio],
        gpt_cond_len=3,
        language="en"
    )
    audio_data = outputs["wav"]
    sf.write(output_file, audio_data, 24000)  # Save audio to file

def generate_audio(text, reference_audio):
    """Split text, process each sentence, and combine into one audio file."""
    sentences = sent_tokenize(text)
    audio_files = []

    # Process each sentence and save as a separate audio file
    for i, sentence in enumerate(sentences):
        output_file = f"/tmp/audio_sentence_{i}.wav"
        process_tts_sentence(sentence, reference_audio, output_file)
        audio_files.append(output_file)

    # Combine audio files with 500ms silence between them
    combined_audio = AudioSegment.empty()
    silence = AudioSegment.silent(duration=500)

    for audio_file in audio_files:
        audio = AudioSegment.from_wav(audio_file)
        combined_audio += audio + silence

    # Save the final combined audio
    final_output = f"/tmp/{uuid.uuid4()}.wav"
    combined_audio.export(final_output, format="wav")

    # Cleanup temporary sentence audio files
    for audio_file in audio_files:
        os.remove(audio_file)

    # Upload combined audio to Google Cloud Storage and get public URL
    audio_url = upload_to_bucket(final_output)

    # Remove the final combined audio file from local storage
    os.remove(final_output)

    return audio_url  # Return public URL of the audio file
