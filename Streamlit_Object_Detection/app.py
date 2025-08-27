import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import os
from ultralytics import YOLO

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YOLO Object Detection")
st.write("Upload an image to detect objects using YOLOv8 model")

@st.cache_resource
def load_yolo_model():
    """Load and cache the YOLO model"""
    try:
        model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for faster inference
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
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

def main():
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
            
            # Load model with spinner
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
                
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            st.write("Please ensure you've uploaded a valid image file (PNG, JPG, or JPEG)")
    else:
        st.info("👆 Upload an image to get started with object detection!")
        
        # Show example
        st.subheader("How it works:")
        st.write("""
        1. **Upload** an image using the file uploader above
        2. **Wait** for the YOLO model to load (first time only)
        3. **View** the original image and detection results side by side
        4. **Explore** the detection details with confidence scores
        """)

if __name__ == "__main__":
    main()