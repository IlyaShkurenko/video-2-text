# import os
# from pydub import AudioSegment
# from nltk.tokenize import sent_tokenize
# import nltk
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# import soundfile as sf

# # Ensure NLTK is downloaded
# nltk.download('punkt')

# # Load configuration and initialize the model
# config = XttsConfig()
# config.load_json("./XTTS-v2/config.json")
# model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="./XTTS-v2/", eval=True)
# model.cuda()

# # Define the path to reference audio files
# file_path = "../myvoice"
# reference_audios = [
#     os.path.join(file_path, "reference_audio1.wav"),
#     os.path.join(file_path, "reference_audio2.wav"),
# ]

# # Function to synthesize speech and save audio
# def process_tts(input_text, speaker_audio, lang, output_file):
#     # Ensure input is not too long for the model
#     if len(input_text) > 250:
#         raise ValueError(f"Input text exceeds 250 tokens: {input_text}")
    
#     outputs = model.synthesize(
#         text=input_text,
#         config=config,
#         speaker_wav=speaker_audio,
#         gpt_cond_len=3,  # Conditional context length
#         language=lang,
#     )
#     audio_data = outputs["wav"]
#     sample_rate = 24000
#     sf.write(output_file, audio_data, sample_rate)
#     print(f"Audio saved: {output_file}")

# # Text for synthesis
# text = "Technology is evolving faster than ever before, changing the way we live and work. Artificial intelligence is at the forefront of this transformation. It helps us solve complex problems, automate routine tasks, and create new opportunities. With tools like XTTS-v2, giving machines a voice has never been easier. Imagine a world where every device can speak in a natural and expressive way. This shift is not just transforming industries but also reshaping our daily lives. From virtual assistants that understand our needs to personalized learning platforms that adapt to individual students, the possibilities are endless. Voice technology is creating more accessible interfaces, breaking down barriers for those with disabilities, and enhancing customer experiences across the globe. As the adoption of AI-powered tools increases, new markets are emerging, driving innovation and economic growth. In healthcare, voice-based systems assist doctors with diagnoses and streamline patient care, while in entertainment, AI-generated voices breathe life into characters and audiobooks. Companies are racing to integrate voice solutions into their products, making interactions more seamless and intuitive."

# # Split text into sentences
# sentences = sent_tokenize(text)

# # List to store temporary audio file paths
# audio_files = []

# # Sequentially synthesize audio for each sentence
# for i, sentence in enumerate(sentences):
#     output_file = f"audio_sentence_{i}.wav"
#     process_tts(sentence, reference_audios, "en", output_file)
#     audio_files.append(output_file)

# # Combine all audio files with silence in between
# combined_audio = AudioSegment.empty()
# silence = AudioSegment.silent(duration=500)  # 500 ms silence between sentences

# for audio_file in audio_files:
#     audio = AudioSegment.from_wav(audio_file)
#     combined_audio += audio + silence

# # Export the combined audio to a single file
# combined_audio.export("combined_audio.wav", format="wav")
# print("Combined audio saved: combined_audio.wav")

# # Remove temporary audio files
# for audio_file in audio_files:
#     os.remove(audio_file)
#     print(f"Removed temporary file: {audio_file}")
