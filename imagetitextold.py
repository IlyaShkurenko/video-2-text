from transformers import pipeline
import torch 

models=['microsoft/kosmos-2-patch14-224', 'adept/fuyu-8b', 'google/pix2struct-textcaps-base', 'Salesforce/blip-image-captioning-base']
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

for model_id in models:
    pipe = pipeline(task="image-text-to-text", model=model_id, device=device)

    outputs = pipe(
        images="https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0001.jpg",
        text="Detailed description of the image, each object and their location, and what they are doing. Location And Objects should be in coordinates, like monkey:(10, 30)",
        max_new_tokens=300
    )

    print(model_id, outputs)
    
# eyError: "Unknown task image-text-to-text, available tasks are ['audio-classification', 'automatic-speech-recognition', 'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 'image-classification', 'image-feature-extraction', 'image-segmentation', 'image-to-image', 'image-to-text', 'mask-generation', 'ner', 'object-detection', 'question-answering', 'sentiment-analysis', 'summarization', 'table-question-answering', 'text-classification', 'text-generation', 'text-to-audio', 'text-to-speech', 'text2text-generation', 'token-classification', 'translation', 'video-classification', 'visual-question-answering', 'vqa', 'zero-shot-audio-classification', 'zero-shot-classification', 'zero-shot-image-classification', 'zero-shot-object-detection', 'translation_XX_to_YY']