import streamlit as st
import os
import tempfile
from openai import OpenAI
import speech_recognition as sr
from deep_translator import GoogleTranslator

# Set up page configuration
st.set_page_config(page_title="Speech-to-Text Transcriber", layout="wide")

# Header
st.title("Speech-to-Text Transcriber")
st.write("""
    Upload an audio file to convert speech into text. Optionally translate the text into another language.
    Supports MP3 and WAV files.
""")

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    transcription_engine = st.selectbox(
        "Transcription Engine",
        ["OpenAI Whisper API", "SpeechRecognition (Local)"]
    )
    st.info("OpenAI Whisper API requires an API key. SpeechRecognition works locally but may be less accurate.")

# Main function for transcription
def transcribe_audio(audio_file, engine):
    if engine == "OpenAI Whisper API":
        if not api_key:
            return None, "API key is required for OpenAI Whisper API"

        try:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = tmp_file.name

            # Use OpenAI Whisper API for transcription
            client = OpenAI(api_key=api_key)
            with open(tmp_path, "rb") as audio_f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_f
                )

            os.remove(tmp_path)
            return response.text, None
        except Exception as e:
            return None, f"Error: {str(e)}"
    else:
        # Use SpeechRecognition library
        try:
            recognizer = sr.Recognizer()

            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = tmp_file.name

            # Use SpeechRecognition for WAV files
            if audio_file.name.lower().endswith('.wav'):
                with sr.AudioFile(tmp_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)

                os.remove(tmp_path)
                return text, None
            else:
                return None, "SpeechRecognition only supports WAV files. For MP3, use the OpenAI Whisper API."
        except sr.UnknownValueError:
            return None, "Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return None, f"Could not request results from Google Speech Recognition service; {e}"
        except Exception as e:
            return None, f"Error: {str(e)}"

# Function for translation
def translate_text(text, target_language):
    try:
        translator = GoogleTranslator(source='auto', target=target_language)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        return f"Translation error: {str(e)}"

# File uploader
st.header("Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

# Language selection for translation
st.header("Translation Options")
translate = st.checkbox("Translate the transcribed text")
target_language = None
if translate:
    languages = {
        "English": "en", "Spanish": "es", "French": "fr",
        "German": "de", "Italian": "it", "Portuguese": "pt",
        "Russian": "ru", "Japanese": "ja", "Chinese (Simplified)": "zh-cn",
        "Arabic": "ar", "Hindi": "hi", "Korean": "ko"
    }
    target_language = st.selectbox("Select target language", list(languages.keys()))
    target_language_code = languages[target_language]

# Process button
if uploaded_file is not None:
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing audio..."):
            transcription, error = transcribe_audio(uploaded_file, transcription_engine)

        if error:
            st.error(error)
        elif transcription:
            # Display transcription
            st.header("Transcription Result")
            st.text_area("Transcribed Text", transcription, height=150)

            # Perform translation if requested
            if translate and target_language:
                with st.spinner(f"Translating to {target_language}..."):
                    translated_text = translate_text(transcription, target_language_code)

                st.header(f"Translated Text ({target_language})")
                st.text_area("Translation", translated_text, height=150)

            # Add download buttons for the results
            st.download_button(
                label="Download Transcription",
                data=transcription,
                file_name="transcription.txt",
                mime="text/plain"
            )

            if translate and target_language:
                st.download_button(
                    label=f"Download {target_language} Translation",
                    data=translated_text,
                    file_name=f"translation_{target_language_code}.txt",
                    mime="text/plain"
                )

# Information about the app
with st.expander("About this app"):
    st.write("""
    ### How it works

    This app converts speech in audio files to text using one of two methods:

    1. **OpenAI Whisper API**: A state-of-the-art speech recognition model with high accuracy, requiring an API key.
    2. **SpeechRecognition**: An open-source library that works with Google's speech recognition service for WAV files.

    ### Supported file formats
    - MP3 (OpenAI Whisper API only)
    - WAV (Both engines)

    ### Translation
    The app can translate the transcribed text to various languages using Google's translation service.
    """)