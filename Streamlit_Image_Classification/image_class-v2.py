import streamlit as st
import torch
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import functional as F
import numpy as np


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


def generate_gradcam(img_tensor, model, target_class=None):
    """Generate Grad-CAM visualization for the predicted class"""
    # Store activations and gradients
    activations = {}
    gradients = {}

    # Function to get activations during forward pass
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output

        return hook

    # Function to get gradients during backward pass
    def save_gradient(name):
        def hook(grad):
            gradients[name] = grad

        return hook

    # Register hooks
    handle_forward = model.layer4.register_forward_hook(save_activation("layer4"))

    # Forward pass
    outputs = model(img_tensor)

    # If no target class is specified, use the predicted class
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()

    # One-hot encode the target class
    target = torch.zeros_like(outputs)
    target[0, target_class] = 1

    # Clear existing gradients
    model.zero_grad()

    # Get activations
    layer4_activation = activations["layer4"]

    # Register hook for gradients
    layer4_activation.register_hook(save_gradient("layer4"))

    # Backward pass
    outputs.backward(target, retain_graph=True)

    # Get gradients
    gradients = gradients["layer4"][0]

    # Global average pooling of gradients
    weights = torch.mean(gradients, dim=(1, 2), keepdim=True)

    # Weighted combination of activation maps
    cam = torch.sum(weights * layer4_activation[0], dim=0)

    # Apply ReLU
    cam = torch.maximum(cam, torch.tensor(0.0))

    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Resize CAM to match the input image size
    cam = cam.detach().cpu().numpy()

    # Clean up
    handle_forward.remove()

    return cam  # Add this code after the "Show Feature Maps" checkbox section


if uploaded_file is not None and model_loaded:
    if st.checkbox("Show Activation Heatmap (Grad-CAM)"):
        st.header("Grad-CAM Visualization")

        with st.spinner("Generating heatmap..."):
            # Preprocess the image
            img_tensor = preprocess(image).unsqueeze(0)

            # Get predictions first to determine the predicted class
            with torch.no_grad():
                outputs = model(img_tensor)

            pred_class = outputs.argmax(dim=1).item()
            pred_class_name = categories[pred_class]

            # Generate Grad-CAM for the predicted class
            cam = generate_gradcam(img_tensor, model, pred_class)

            # Create a figure with two subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            img_np = np.array(image.resize((224, 224)))
            ax1.imshow(img_np)
            ax1.set_title("Original Image")
            ax1.axis("off")

            # Heatmap
            ax2.imshow(cam, cmap="jet")
            ax2.set_title("Activation Heatmap")
            ax2.axis("off")

            # Overlay heatmap on original image
            import cv2

            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = np.uint8(255 * cam_resized)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Convert RGB to BGR for OpenCV
            img_np = (
                img_np[:, :, ::-1].copy()
                if img_np.shape[-1] == 3
                else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            )

            # Superimpose the heatmap on original image
            superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
            superimposed_img = superimposed_img[:, :, ::-1]  # Convert back to RGB

            ax3.imshow(superimposed_img)
            ax3.set_title(f"Attention for: {pred_class_name}")
            ax3.axis("off")

            st.pyplot(fig)

            st.markdown(
                f"""
            **Grad-CAM Visualization Explanation:**

            This visualization shows where the model is focusing to make its prediction. 
            Bright areas in the heatmap (red/yellow) are regions that strongly influence 
            the classification decision for "{pred_class_name}".
            """
            )  # App information
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
