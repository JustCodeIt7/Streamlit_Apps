import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import os
from ultralytics import YOLO
from transformers import pipeline

st.set_page_config(
    page_title="Object Detection/Classification",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Object Detection/Classification")
st.write("Upload an image to detect objects using YOLOv8 or classify using CLIP")

@st.cache_resource
def load_yolo_model():
    """Load and cache the YOLO model"""
    try:
        model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for faster inference
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None

@st.cache_resource
def load_clip_model():
    """Load and cache the CLIP pipeline for zero-shot image classification"""
    try:
        pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
        return pipe
    except Exception as e:
        st.error(f"Error loading CLIP model: {str(e)}")
        return None

def process_image(image, model):
    """Process image through YOLO model and return annotated image"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run inference
        results = model(img_array)
        
        # Get the annotated image from YOLO results
        annotated_img = results[0].plot()
        
        # Convert back to PIL Image
        annotated_pil = Image.fromarray(annotated_img)
        
        return annotated_pil, results[0]
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def process_image_clip(image, pipe):
    """Process image through CLIP pipeline for zero-shot classification"""
    try:
        # Define text prompts for common objects/classes (zero-shot)
        candidate_labels = [
            "a photo of a person",
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a car",
            "a photo of a house",
            "a photo of food",
            "a photo of a landscape",
            "a photo of furniture",
            "a photo of electronics",
            "a photo of animals"
        ]
        
        # Run classification
        results = pipe(image, candidate_labels=candidate_labels)
        
        # Format results
        formatted_results = []
        for res in results:
            formatted_results.append({
                'Class': res['label'],
                'Probability': f"{res['score']:.2%}"
            })
        
        return formatted_results
    except Exception as e:
        st.error(f"Error processing image with CLIP: {str(e)}")
        return None

def main():
    # Sidebar for model selection
    model_choice = st.sidebar.selectbox(
        "Select Model for Object Detection/Classification",
        ["YOLOv8", "CLIP"],
        help="YOLOv8 performs object detection with bounding boxes. CLIP performs zero-shot image classification."
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        try:
            # Validate and load image
            image = Image.open(uploaded_file)
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            if model_choice == "YOLOv8":
                # Load YOLO model with spinner
                with st.spinner("Loading YOLO model..."):
                    model = load_yolo_model()
                
                if model is not None:
                    # Process image with spinner
                    with st.spinner("Detecting objects..."):
                        annotated_image, results = process_image(image, model)
                    
                    if annotated_image is not None:
                        with col2:
                            st.subheader("Object Detection Results")
                            st.image(annotated_image, use_column_width=True)
                        
                        # Display detection details
                        st.subheader("Detection Details")
                        
                        if len(results.boxes) > 0:
                            # Create detection summary
                            detections = []
                            for box in results.boxes:
                                class_id = int(box.cls[0])
                                confidence = float(box.conf[0])
                                class_name = model.names[class_id]
                                detections.append({
                                    'Class': class_name,
                                    'Confidence': f"{confidence:.2%}"
                                })
                            
                            # Display as table
                            st.dataframe(detections, use_container_width=True)
                            
                            # Summary stats
                            unique_classes = set([d['Class'] for d in detections])
                            st.info(f"Detected {len(detections)} objects across {len(unique_classes)} different classes")
                        else:
                            st.info("No objects detected in the image")
                else:
                    st.error("Failed to load YOLO model. Please check your internet connection.")
            
            elif model_choice == "CLIP":
                # Load CLIP pipeline with spinner
                with st.spinner("Loading CLIP model..."):
                    pipe = load_clip_model()
                
                if pipe is not None:
                    # Process image with spinner
                    with st.spinner("Classifying image..."):
                        results = process_image_clip(image, pipe)
                    
                    if results is not None:
                        with col2:
                            st.subheader("Classification Results")
                            st.dataframe(results, use_container_width=True)
                        
                        # Summary
                        top_class = results[0]['Class'] if results else "N/A"
                        top_prob = results[0]['Probability'] if results else "N/A"
                        st.info(f"Top prediction: {top_class} with {top_prob} confidence")
                else:
                    st.error("Failed to load CLIP model. Please check your internet connection.")
                
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            st.write("Please ensure you've uploaded a valid image file (PNG, JPG, or JPEG)")
    else:
        st.info("👆 Upload an image to get started with object detection/classification!")
        
        # Show example
        st.subheader("How it works:")
        st.write("""
        1. **Select Model:** Choose YOLOv8 for object detection (with bounding boxes) or CLIP for zero-shot image classification.
        2. **Upload** an image using the file uploader above
        3. **Wait** for the selected model to load (first time only)
        4. **View** the original image and results side by side
        5. **Explore** the detection/classification details
        """)

if __name__ == "__main__":
    main()
