import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

images = ["https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0001.jpg", "https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0009.png"]

for url in images:
    image = Image.open(requests.get(url, stream=True).raw)
    question = 'Detailed description of the image without unnecessary adjectives or embellishments. Include each object, animal and their location, and what they are doing. Use structured format and always mention the location of the objects or animals'
    msgs = [{'role': 'user', 'content': question}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
    )
    print(res)