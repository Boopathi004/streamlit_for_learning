import streamlit as st
from chatbot import get_response

st.title("🤖 NLP Chatbot")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input
user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        response = get_response(user_input)

        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", response))

# Display chat
for sender, msg in st.session_state.messages:
    if sender == "You":
        st.write(f"🧑 {msg}")
    else:
        st.write(f"🤖 {msg}")