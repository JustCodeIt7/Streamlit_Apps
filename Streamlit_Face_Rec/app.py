import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Facial Recognition App", page_icon="👤", layout="wide")

# Function to load known faces and their encodings
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        return known_face_encodings, known_face_names

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Load image file
            image_path = os.path.join(known_faces_dir, filename)
            face_image = face_recognition.load_image_file(image_path)

            # Get face encoding - assume each file has one face
            try:
                face_encoding = face_recognition.face_encodings(face_image)[0]
                # Add encoding to list
                known_face_encodings.append(face_encoding)
                # Use filename (without extension) as the name
                known_face_names.append(os.path.splitext(filename)[0])
            except IndexError:
                st.warning(f"No face found in {filename}. Skipping.")

    return known_face_encodings, known_face_names

def process_image(image, known_face_encodings, known_face_names):
    # Convert to numpy array if it's not already
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert from BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image

    # Find all face locations and face encodings
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Create a copy of the image to draw on
    image_to_draw = np.copy(image)

    # List to store recognized names
    recognized_names = []

    # Loop through each face found
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        recognized_names.append(name)

        # Draw a rectangle around the face
        cv2.rectangle(image_to_draw, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(image_to_draw, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_to_draw, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    return image_to_draw, recognized_names

def main():
    st.title("Facial Recognition App")

    # Create a directory for known faces if it doesn't exist
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Add Known Faces")

        # Option to add a new face
        new_face_name = st.text_input("Name for the new face")
        add_face_method = st.radio(
            "Add face using:",
            ["Upload Image", "Capture from Camera"]
        )

        if add_face_method == "Upload Image":
            upload_image = st.file_uploader("Upload a clear face image", type=["jpg", "jpeg", "png"])

            if upload_image is not None and new_face_name:
                # Save the uploaded image
                image_path = os.path.join(known_faces_dir, f"{new_face_name}.jpg")
                with open(image_path, "wb") as f:
                    f.write(upload_image.getbuffer())
                st.success(f"Added {new_face_name} to known faces!")

        else:  # Capture from Camera
            st.write("Take a photo to add as a known face:")
            face_capture = st.camera_input("Capture Face")

            if face_capture is not None and new_face_name:
                # Save the captured image
                image_path = os.path.join(known_faces_dir, f"{new_face_name}.jpg")
                with open(image_path, "wb") as f:
                    f.write(face_capture.getbuffer())
                st.success(f"Added {new_face_name} to known faces!")

        # Display known faces
        known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

        if len(known_face_names) == 0:
            st.info("No known faces found. Please add faces using the controls above.")
        else:
            st.subheader("Known Faces")
            for name in known_face_names:
                st.write(f"- {name}")

    with col2:
        st.header("Facial Recognition")
        st.write("Use your camera to recognize faces")

        # Camera input for face recognition
        camera_image = st.camera_input("Take a photo to recognize faces")

        # Load known faces for recognition
        known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

        # Process the image when available
        if camera_image is not None:
            # Check if we have known faces to compare with
            if len(known_face_encodings) == 0:
                st.warning("No known faces available for recognition. Please add faces first.")
            else:
                # Convert to PIL Image
                image = Image.open(camera_image)

                # Process the image and recognize faces
                with st.spinner("Recognizing faces..."):
                    result_image, recognized_names = process_image(np.array(image), known_face_encodings, known_face_names)

                # Display the processed image
                st.image(result_image, caption="Recognition Results", use_column_width=True)

                # Display recognition results
                if recognized_names:
                    st.subheader("People Recognized")
                    for name in set(recognized_names):
                        st.write(f"- {name}")

                    # Show summary statistics
                    st.write(f"Total faces detected: {len(recognized_names)}")
                    st.write(f"Known faces: {len(recognized_names) - recognized_names.count('Unknown')}")
                    st.write(f"Unknown faces: {recognized_names.count('Unknown')}")
                else:
                    st.write("No faces detected in the image.")

    # Instructions section
    st.markdown("---")
    st.subheader("How to Use This App")
    st.write("""
    1. First, add known faces using the left panel. You can upload images or take photos with your camera.
    2. Each person should have a clear, well-lit frontal face image.
    3. Once you've added known faces, use the camera on the right to take a photo for recognition.
    4. The app will identify known faces and mark unknown faces as "Unknown".
    """)

    # Information about the app
    st.markdown("---")
    st.write("This app uses the face_recognition library which is built on dlib's state-of-the-art face recognition algorithms.")
    st.caption("Note: For best results, ensure good lighting and clear facial visibility.")

# Run the app
if __name__ == "__main__":
    main()