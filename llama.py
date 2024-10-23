import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
processor = AutoProcessor.from_pretrained(model_id)

images = ["https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0001.jpg", "https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0009.png"]

for url in images:
    image = Image.open(requests.get(url, stream=True).raw)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Detailed description of the image, each object, animal and their location, and what they are doing. Use structured format and always mention the location of the objects or animals"}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=300)
    print(processor.decode(output[0]))
