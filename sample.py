from groq import Groq
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class GroqChatClient:
    def __init__(self, model_id='llama-3.3-70b-versatile', system_message=None, api_key=None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("API key is not provided. Please set GROQ_API_KEY as an environment variable or pass it explicitly.")
        self.client = Groq(api_key=api_key)
        self.model_id = model_id
        self.messages = []
        if system_message:
            self.messages.append({'role': 'system', 'content': system_message})
        
    def draft_message(self, prompt, role='user'):
        return {'role': role, 'content': prompt}
        
    def send_request(self, message, temperature=0.5, max_tokens=1024, stream=False, stop=None):
        self.messages.append(message)
        try:
            chat_completion = self.client.chat.completions.create(
                messages=self.messages,
                model=self.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stop=stop
            )
            print(chat_completion)  # Debug
            if not stream:
                response = {
                    'content': chat_completion.choices[0].message.content,
                    'role': chat_completion.choices[0].message.role,
                }
                self.messages.append(self.draft_message(response['content'], response['role']))
                return response
            return chat_completion
        except Exception as e:
            raise ValueError(f"Error with Groq API: {e}")




# Streamlit App
st.title("LegalEase: AI Legal Assistant")

system_message = (
    "You are an AI legal assistant trained in handling civil cases in the Philippines."
    "Questions outside this area will not be addressed."
)

#api_key = os.getenv("GROQ_API_KEY")
api_key = st.secrets["GROQ_API_KEY"]
if not api_key:
    st.error("API key is not set. Please set it as 'GROQ_API_KEY' environment variable.")
else:
    client = GroqChatClient(system_message=system_message, api_key=api_key)
    if 'messages' not in st.session_state:
        st.session_state.messages = client.messages

    user_input = st.chat_input("Enter your inquiry:")
    
    if user_input:
        user_message = client.draft_message(user_input)
        try:
            response = client.send_request(user_message)
            st.session_state.messages.append(user_message)
            assistant_message = client.draft_message(response['content'], 'assistant')
            st.session_state.messages.append(assistant_message)

            for message in st.session_state.messages:
                role = message['role']
                content = message['content']
                if role == 'user':
                    st.chat_message("user").markdown(content)
                elif role == 'assistant':
                    st.chat_message("assistant").markdown(content)
        except Exception as e:
            st.error(f"Error: {e}")

