from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from IPython.display import Audio
from scipy.io.wavfile import write
import os

config = XttsConfig()
config.load_json("./XTTS-v2/config.json")

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./XTTS-v2/")
model.cuda() 

text_to_speak = "Technology is evolving faster than ever before, changing the way we live and work. Artificial intelligence is at the forefront of this transformation. It helps us solve complex problems, automate routine tasks, and create new opportunities. With tools like XTTS-v2, giving machines a voice has never been easier. Imagine a world where every device can speak in a natural and expressive way."

file_path = "../momvoice"
reference_audios = [
    os.path.join(file_path, "reference_audio1.wav"),
    os.path.join(file_path, "reference_audio2.wav"),
    # os.path.join(file_path, "reference_audio3.wav")
]

outputs = model.synthesize(
    text=text_to_speak,
    config=config,
    speaker_wav=reference_audios,
    gpt_cond_len=5,
    language="en"
)

Audio(data=outputs['wav'], rate=24000)

output_file_path = './output.wav'
write(output_file_path, 24000, outputs['wav'])

print(f"output: {output_file_path}")
