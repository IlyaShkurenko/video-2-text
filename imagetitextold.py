from transformers import pipeline
import torch 

models=['HuggingFaceM4/idefics2-8b', 'adept/fuyu-8b']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_id in models:
    pipe = pipeline(task="image-text-to-text", model=model_id, device=device)

    outputs = pipe(
        images="https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0001.jpg",
        text="Detailed description of the image, each object and their location, and what they are doing. Location And Objects should be in coordinates, like monkey:(10, 30)",
        max_new_tokens=300
    )
	
    print(outputs)