import replicate

input = {
    "image": "https://replicate.delivery/pbxt/HrXsgowfhbZi3dImGZoIcvnz7oZfMtFY4UAEU8vBIakTd8JQ/watercolour-4799014_960_720.jpg",
    "clip_model_name": "ViT-L/14"
}

output = replicate.run(
    "pharmapsychotic/clip-interrogator:8151e1c9f47e696fa316146a2e35812ccf79cfc9eba05b11c7f450155102af70",
    input=input
)
print(output)