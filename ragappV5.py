from groq import Groq
import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import BaseLLM
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field

# === CONFIG ===
PDF_FILES = [
    ("ra386_books/RA386_Book1.pdf", "Persons"),
    ("ra386_books/RA386_Book2.pdf", "Property"),
    ("ra386_books/RA386_Book3.pdf", "Succession"),
    ("ra386_books/RA386_Book4.pdf", "Obligations & Contracts"),
]
FAISS_INDEX_PATH = "faiss_index_multi"

# === 1. Custom Groq LLM ===
class GroqLLM(BaseLLM):
    def __init__(self, model_id='llama-3-70b-8192', api_key=None):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key is not provided.")
        self.client = Groq(api_key=self.api_key)

    def _call(self, prompt: str, stop: list = None) -> str:
        response = self.client.send_request({'role': 'user', 'content': prompt}, stop=stop)
        return response['content']

    def _generate(self, prompts: list, stop: list = None) -> list:
        return [self._call(prompt, stop) for prompt in prompts]

    @property
    def _llm_type(self) -> str:
        return "groq"

# === 2. Load and Vectorize Multiple PDFs ===
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={"device": "cpu"}
    )

    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

    all_documents = []
    for file_path, book_label in PDF_FILES:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Attach metadata (book_label) to each document
        for doc in documents:
            doc.metadata['book'] = book_label
        all_documents.extend(documents)

    # Split documents carefully
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_documents)

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

# === 3. Custom Retriever that Filters by Book(s) ===
class FilteredRetriever(BaseRetriever):
    retrievers: Dict[str, BaseRetriever] = Field(...)
    default_retriever: BaseRetriever = Field(...)

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        return await self.get_relevant_documents(query, **kwargs)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        lowered = query.lower()
        if "succession" in lowered:
            retriever = self.retrievers.get("Succession", self.default_retriever)
        elif "property" in lowered:
            retriever = self.retrievers.get("Property", self.default_retriever)
        elif "persons" in lowered or "person" in lowered:
            retriever = self.retrievers.get("Persons", self.default_retriever)
        elif "obligations" in lowered or "contracts" in lowered:
            retriever = self.retrievers.get("Obligations & Contracts", self.default_retriever)
        else:
            retriever = self.default_retriever

        return retriever.get_relevant_documents(query, **kwargs)

# === 4. Build retrievers ===
vectorstore = get_vectorstore()

# Create specific retrievers per book
retrievers_by_book = {}
for book in ["Persons", "Property", "Succession", "Obligations & Contracts"]:
    retrievers_by_book[book] = vectorstore.as_retriever(
        search_kwargs={"filter": {"book": book}}
    )

default_retriever = vectorstore.as_retriever()

retriever = FilteredRetriever(
    retrievers=retrievers_by_book,
    default_retriever=default_retriever
)

# === 5. LLM Setup ===
groq_llm = ChatGroq(
    model="llama3-8b-8192",
    #api_key=os.environ.get("GROQ_API_KEY"),
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0
)

custom_prompt = PromptTemplate.from_template("""
You are a legal expert specializing in Philippine civil law, particularly mobile property, real estate, succession, obligations, and contracts.

Use only the provided context to answer. If the context is insufficient, say you don't know.
Cite relevant articles when possible.

Summaries:
{summaries}

Question:
{question}

Helpful, accurate legal answer in Markdown:
""")

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=groq_llm,
    retriever=retriever,
    chain_type='stuff',
    chain_type_kwargs={"prompt": custom_prompt}
)

# === 6. Streamlit App ===
st.title("⚖️ LegalEase: Civil Law Assistant (Multi-Book Filtering)")
st.info("Ask about civil cases involving persons, property, succession, or obligations under Philippine law.")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("API key is not set. Please check your environment variables.")
else:
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # ✅ Display the chat history first
    for message in st.session_state.messages:
        role = message['role']
        content = message['content']
        st.chat_message(role).markdown(content)

    # 🧠 Handle new input
    user_input = st.chat_input("Enter your legal question:")

    if user_input:
        user_message = {'role': 'user', 'content': user_input}
        st.session_state.messages.append(user_message)

        try:
            response = chain.invoke({"question": user_input})
            assistant_message = {'role': 'assistant', 'content': response['answer']}
            st.session_state.messages.append(assistant_message)

            # The message will be shown automatically in the next rerun via the history loop
            if response.get('sources'):
                st.markdown("**Sources:**")
                for source in response['sources'].split(", "):
                    st.markdown(f"- {source}")

        except Exception as e:
            st.error(f"Error: {e}")

st.caption("© 2025 LegalEase. General legal information only. Not a substitute for professional advice.")
