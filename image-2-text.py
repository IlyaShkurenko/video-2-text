import requests
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

device = torch.device("cuda")

img_urls = [
    "https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0001.jpg",
]

model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device)

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

images = [Image.open(requests.get(img_urls[0], stream=True).raw)]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Detailed description of the image, each object and their location, and what they are doing. Location And Objects should be in coordinates, like monkey:(10, 30)"},
        ]
    },
]