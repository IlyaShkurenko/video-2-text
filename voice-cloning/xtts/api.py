from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
import os

file_path = "../momvoice"
reference_audios = [
    os.path.join(file_path, "reference_audio1.wav"),
    os.path.join(file_path, "reference_audio2.wav"),
    # os.path.join(file_path, "reference_audio3.wav")
]

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="Technology is evolving faster than ever before, changing the way we live and work. Artificial intelligence is at the forefront of this transformation. It helps us solve complex problems, automate routine tasks, and create new opportunities. With tools like XTTS-v2, giving machines a voice has never been easier. Imagine a world where every device can speak in a natural and expressive way.",
                file_path="output2.wav",
                speaker_wav=reference_audios,
                language="en")