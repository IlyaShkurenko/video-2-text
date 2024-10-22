import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

prompt = "Detailed description of the image, each object, animal and their location, and what they are doing. Use structured format and always mention the location of the objects or animals"

images = ["https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0001.jpg", "https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0009.png"]

for url in images:
	image = Image.open(requests.get(url, stream=True).raw)

	inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

	generated_ids = model.generate(
				input_ids=inputs["input_ids"],
				pixel_values=inputs["pixel_values"],
				max_new_tokens=1024,
				num_beams=3,
				do_sample=False
	)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

	parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

	print(parsed_answer)