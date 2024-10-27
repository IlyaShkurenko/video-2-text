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

def sample_frames(url, num_frames):
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
            {"type": "text", "text": "On which second grill appears"},
            {"type": "video"},
            ],
    },
]
print('conversation', conversation)
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print('prompt',prompt)

video_url = "https://storage.googleapis.com/tidy-federation-332618.appspot.com/video/dragon.MOV"
video = sample_frames(video_url, 8)
print('video', video)

inputs = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)
print('inputs', inputs)
output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
# print('output', output)
print('output', processor.decode(output[0][2:], skip_special_tokens=True))
