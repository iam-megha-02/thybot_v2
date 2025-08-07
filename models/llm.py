# models/llm.py

from groq import Groq
from config.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def get_groq_model():
    class GroqWrapper:
        def invoke(self, prompt_or_messages):
            # If plain string, convert to message format
            if isinstance(prompt_or_messages, str):
                prompt_or_messages = [{"role": "user", "content": prompt_or_messages}]
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=prompt_or_messages
            )
            return response.choices[0].message.content
    return GroqWrapper()
