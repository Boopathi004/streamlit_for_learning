# pip install SpeechRecognition
# pip install pyttsx3
# pip install pypiwin32
# pip install pipwin
# pip install gtts

import streamlit as st
import speech_recognition as sr
from chatbot import get_response
from gtts import gTTS
import tempfile
import os

# gTTS (Google Text-to-Speech), a Python library and CLI tool to interface with Google Translate's text-to-speech API.

# speech_recognition.Recognizer is the primary class in the SpeechRecognition Python library used to process and transcribe audio. It acts as a central hub for managing audio settings and interfacing with various speech-to-text APIs

st.set_page_config(page_title="Voice Chatbot", layout="centered")

st.title("Voice Chatbot")

# Initialize recognizer
recognizer = sr.Recognizer()

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
#  Speech to Text
# ----------------------------
def transcribe_audio(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            return "Sorry, could not understand."

# ----------------------------
#  Text to Speech (browser)
# ----------------------------
def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        return tmp.name

# ----------------------------
#  Display Chat Messages
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ----------------------------
#  Text Input
# ----------------------------
user_input = st.chat_input("Type your message...")

# ----------------------------
#  Voice Input
# ----------------------------
audio = st.audio_input("Or speak here 🎤")

if audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio.read())
        tmp_path = tmp.name

    user_input = transcribe_audio(tmp_path)

# ----------------------------
#  Process Input
# ----------------------------
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get bot response
    bot_reply = get_response(user_input)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Display bot message
    with st.chat_message("assistant"):
        st.write(bot_reply)

        # 🔊 Play audio
        audio_file = speak(bot_reply)
        st.audio(audio_file)

        # Cleanup (optional)
        try:
            os.remove(audio_file)
        except:
            pass


