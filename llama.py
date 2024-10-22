import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0001.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(image)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Detailed description of the image, each object and their location, and what they are doing. Location And Objects should be in coordinates, like monkey:(10, 30)"}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
print(input_text)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)
print(inputs)

output = model.generate(**inputs, max_new_tokens=300)
print(processor.decode(output[0]))
