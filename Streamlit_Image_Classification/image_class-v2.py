import streamlit as st
import torch
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import functional as F


# Caching the model loading so it doesn't reload on every run
@st.cache_resource
def load_model():
    # Load pre-trained ResNet50 model
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.eval()  # Set the model to evaluation mode
    return model, weights


# Function to get feature maps from intermediate layers
def get_activation_maps(img_tensor, model):
    # Create a hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations["layer4"] = output

    # Register the hook on the last layer (layer4 in ResNet50)
    model.layer4.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)

    return activations


# Title and description
st.title("🖼️ Image Classification with PyTorch")
st.markdown(
    """
This app uses a pre-trained ResNet50 model to classify images. 
Upload an image to see what the model thinks it is, along with class probabilities and feature maps.
"""
)

# Load model
try:
    model, weights = load_model()
    # Get the image transformations from the model weights
    preprocess = weights.transforms()
    # Get class mapping
    categories = weights.meta["categories"]
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# Create columns for layout
col1, col2 = st.columns([1, 1])

# Image upload in the left column
with col1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image with PIL
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Model prediction in the right column
with col2:
    if uploaded_file is not None and model_loaded:
        st.header("Classification Results")

        with st.spinner("Classifying..."):
            # Preprocess the image
            img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

            # Get predictions
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]

            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities, 5)

            # Display predictions
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                class_name = categories[idx]
                confidence = prob.item() * 100
                st.write(f"**{i+1}. {class_name}**: {confidence:.2f}%")

            # Create a bar chart for top 5 predictions
            st.subheader("Probability Distribution")

            # Convert top class names and probabilities to a DataFrame
            df = pd.DataFrame(
                {"Probability": top_probs.numpy() * 100},
                index=[categories[idx] for idx in top_indices],
            )

            # Create the bar chart
            st.bar_chart(df)

# Add feature map visualization below
if uploaded_file is not None and model_loaded:
    if st.checkbox("Show Feature Maps"):
        st.header("Feature Maps Visualization")

        with st.spinner("Generating feature maps..."):
            # Get activation maps
            img_tensor = preprocess(image).unsqueeze(0)
            activations = get_activation_maps(img_tensor, model)

            # Get the activation maps
            feature_maps = activations["layer4"].squeeze(0).cpu().numpy()

            # Display a sample of feature maps (first 16)
            num_maps = min(16, feature_maps.shape[0])

            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            for i, ax in enumerate(axes.flat):
                if i < num_maps:
                    ax.imshow(feature_maps[i], cmap="viridis")
                    ax.set_title(f"Map {i+1}")
                    ax.axis("off")
                else:
                    ax.axis("off")

            st.pyplot(fig)

# App information
with st.expander("About this app"):
    st.markdown(
        """
    This app uses a pre-trained ResNet50 model trained on ImageNet.

    **How it works:**
    1. Upload an image
    2. The image is preprocessed (resized, normalized)
    3. The model returns probabilities for 1,000 different classes
    4. The app displays the top 5 most likely classes

    Feature maps show the activations from the last convolutional layer, visualizing what patterns the model detects in your image.
    """
    )
