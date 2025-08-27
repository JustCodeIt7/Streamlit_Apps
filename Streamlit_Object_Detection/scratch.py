# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
pipe(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png",
    candidate_labels=["animals", "humans", "landscape"],
)
