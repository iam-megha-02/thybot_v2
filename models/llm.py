# models/llm.py

import streamlit as st
from groq import Groq

# Load GROQ API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def get_groq_model():
    class GroqWrapper:
        def invoke(self, prompt_or_messages):
            # Convert string to message format if needed
            if isinstance(prompt_or_messages, str):
                prompt_or_messages = [{"role": "user", "content": prompt_or_messages}]
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=prompt_or_messages
            )
            return response.choices[0].message.content
    return GroqWrapper()
