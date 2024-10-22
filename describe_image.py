import replicate

input = {
    "image": "https://storage.googleapis.com/tidy-federation-332618.appspot.com/img/frame_0009.png",
    "clip_model_name": "ViT-L-14/openai"
}

output = replicate.run(
    "pharmapsychotic/clip-interrogator:8151e1c9f47e696fa316146a2e35812ccf79cfc9eba05b11c7f450155102af70",
    input=input
)
print(output)