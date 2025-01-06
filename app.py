from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from groq import Groq
import streamlit as st
import os

from wxai_langchain.llm import LangChainInterface

'''creds = {
    'apikey': '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',
    'url': 'https://jp-tok.ml.cloud.ibm.com'
}'''

creds = {
    'apikey':'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',
    'url': 'https://api.groq.com/openai/v1/chat/completions'
}

#client = Groq()

'''llm = ChatGroq(
    credentials=creds,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)'''

llm = ChatGroq(
    credentials = creds,
    model = 'llama-3.3-70b-versatile',
    params = {
        'decoding_method':'sample',
        'max_new_tokens':200,
        'temperature':0.05
    }
)

st.title('LegalEase')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
prompt = st.chat_input('Pass Your Prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    response = llm(prompt)
    st.chat_message('assistant').markdown(response)
    st.session_state.message.append(
        {'role':'assistant',
         'content':response}
    )