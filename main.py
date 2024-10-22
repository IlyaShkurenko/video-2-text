# from tkinter import Image
import os
import uuid
import requests
from PIL import Image
import cv2
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)
# def download(url: str, dest_folder: str):
#     if not os.path.exists(dest_folder):
#         os.makedirs(dest_folder)  # create folder if it does not exist

#     filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
#     file_path = os.path.join(dest_folder, filename)

#     r = requests.get(url, stream=True)
#     if r.ok:
#         print("saving to", os.path.abspath(file_path))
#         with open(file_path, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=1024 * 8):
#                 if chunk:
#                     f.write(chunk)
#                     f.flush()
#                     os.fsync(f.fileno())
#     else:  # HTTP status code 4XX/5XX
#         print("Download failed: status code {}\n{}".format(r.status_code, r.text))

def sample_frames(url, num_frames):
    # path_id = str(uuid.uuid4())
    # path = f"./{path_id}.mp4"
    
    # download(url, path)

    # file_size = os.path.getsize(path)
    # print(f"Video saved to {path}, size: {file_size} bytes")

    # with open(path, "wb") as f:
    #      f.write(response.content)

    video = cv2.VideoCapture(url)
    print(video)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            continue
        if i % interval == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_img)
    print(frames)
    video.release()
    print(1)
    return frames

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Carefully identify every object present in the video and determine their location, as it is essential to understand what changes over time. Provide a detailed summary of what happens in the video, without unnecessary adjectives or embellishments. Your goal is to determine actions and objects in the video."},
            {"type": "video"},
            ],
    },
]
print('conversation', conversation)
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print('prompt',prompt)

video_url = "https://storage.googleapis.com/tidy-federation-332618.appspot.com/video/balivaran.mp4"
video = sample_frames(video_url, 8)
print('video', video)

inputs = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)
print('inputs', inputs)
output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
# print('output', output)
print('output', processor.decode(output[0][2:], skip_special_tokens=True))
