# from tkinter import Image
import uuid
import requests
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
    response = requests.get(url)
    print(response)
    path_id = str(uuid.uuid4())
    path = f"./{path_id}.mp4"

    with open(path, "wb") as f:
         f.write(response.content)

    video = cv2.VideoCapture(path)
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
            {"type": "text", "text": "Describe each second on the video"},
            {"type": "video"},
            ],
    },
]
print(2)
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(3)

video_url = "https://www.dropbox.com/scl/fi/yj51cd30nkokbu6s47g21/IMG_6584.MOV?rlkey=tpqt90ze101etaiwzcez6igdr&st=2yq821dx&dl=0"
video = sample_frames(video, 8)
print(4)

print(5)
inputs = processor(text=prompt, videos=video, padding=True, return_tensors="pt").to(model.device)
print(6)
output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(output)
print(processor.decode(output[0][2:], skip_special_tokens=True))
print(8)
