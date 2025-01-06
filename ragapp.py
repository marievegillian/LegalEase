from groq import Groq
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import BaseLLM

# Custom Groq LLM Wrapper
class GroqLLM(BaseLLM):
    def __init__(self, model_id='llama-3.3-70b-versatile', api_key=None):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key is not provided. Please set GROQ_API_KEY as an environment variable or pass it explicitly.")
        self.client = Groq(api_key=self.api_key)

    def _call(self, prompt: str, stop: list = None) -> str:
        # Make request to Groq API and return the result
        response = self.client.send_request(
            {'role': 'user', 'content': prompt},
            stop=stop
        )
        return response['content']

    # Implementation of the abstract methods
    def _generate(self, prompts: list, stop: list = None) -> list:
        # This method is used by Langchain's LLMs for batch generation.
        return [self._call(prompt, stop) for prompt in prompts]

    @property
    def _llm_type(self) -> str:
        # This defines the type of the LLM.
        return "groq"

# RAG part: Load the PDF and create the vector store index
@st.cache_resource
def load_pdf():
    pdf_name = 'RA 386.pdf'
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    return index

# Initialize the PDF index
index = load_pdf()

# Initialize the custom Groq LLM
groq_llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))

# Set up the RetrievalQA chain with the GroqLLM
chain = RetrievalQA.from_chain_type(
    llm=groq_llm,  # Using the custom Groq LLM
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question'
)

#Streamlit App
st.title("LegalEase: AI Legal Assistant")

system_message = (
    "You are a legal assistant with 15 years of experience in handling civil cases in the Philippines. "
    "Questions outside this area will not be addressed."
)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("API key is not set. Please set it as 'GROQ_API_KEY' environment variable.")
else:
    # Initialize GroqChatClient if API key is available
    if 'messages' not in  st.session_state:
        st.session_state.messages = []
        
    # User input for query
    user_input = st.chat_input("Enter your inquiry:")
    
    if user_input:
        user_message = {'role': 'user', 'content': user_input}
        try:
            # Get the response from the chain
            response = chain.run(user_input)
            
            # Append user and assistant messages to session state
            st.session_state.messages.append(user_message)
            assistant_message = {'role': 'assistant', 'content': response}
            st.session_state.messages.append(assistant_message)

            # Display the conversation
            for message in st.session_state.messages:
                role = message['role']
                content = message['content']
                if role == 'user':
                    st.chat_message("user").markdown(content)
                elif role == 'assistant':
                    st.chat_message("assistant").markdown(content)
        except Exception as e:
            st.error(f"Error: {e}")
