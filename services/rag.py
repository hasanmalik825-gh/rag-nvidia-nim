from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from starlette.datastructures import UploadFile as FastAPIUploadFile
from streamlit.runtime.uploaded_file_manager import UploadedFile as StreamlitUploadedFile
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank, ChatNVIDIA
import tempfile
from typing import List, Union
import os
from utils.document_comparison import _get_string_hash
from constants import VECTOR_STORE_PATH, NVIDIA_API_KEY

def load_document(file: Union[FastAPIUploadFile, StreamlitUploadedFile]) -> List[Document]:
    """
    Handles file loading for both FastAPI's UploadFile and Streamlit's UploadedFile.

    Args:
        file: FastAPI's UploadFile or Streamlit's UploadedFile.

    Returns:
        Loaded document.
    """
    # Check file type
    if isinstance(file, FastAPIUploadFile):
        # FastAPI UploadFile handling
        file_name = file.filename
        file_content = file.file.read()
    elif isinstance(file, StreamlitUploadedFile):
        # Streamlit UploadedFile handling
        file_name = file.name
        file_content = file.read()
    else:
        raise ValueError("Unsupported file type: Must be FastAPI's UploadFile or Streamlit's UploadedFile.")

    # Create a temporary file for processing
    file_extension = file_name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    # Determine the loader based on file type
    if file_extension == "pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == "txt":
        loader = TextLoader(temp_file_path)
    else:
        os.remove(temp_file_path)  # Cleanup
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        # Load the document
        document = loader.load()
    finally:
        # Ensure temporary file is cleaned up
        os.remove(temp_file_path)

    return document

def split_document(documents: List[Document]) -> List[Document]:
    """
    This function is used to split the documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# For this open-source option if someone intend to use it.
def embedder_by_nvidia(api_key: str = NVIDIA_API_KEY) -> NVIDIAEmbeddings:
    """
    This function is used to embed the documents using huggingface embedding model.
    """
    embedder = NVIDIAEmbeddings(nvidia_api_key=api_key)
    return embedder

def reranker_by_nvidia(api_key: str = NVIDIA_API_KEY) -> NVIDIARerank:
    """
    This function is used to rerank the documents using nvidia reranker model.
    """
    reranker = NVIDIARerank(nvidia_api_key=api_key)
    return reranker

def llm_by_nvidia(
        model: str,
        api_key: str = NVIDIA_API_KEY
    ) -> ChatNVIDIA:
    """
    This function is used to create a llm using nvidia llm model.
    """
    llm = ChatNVIDIA(model=model, nvidia_api_key=api_key)
    return llm

def create_or_load_vector_store(
        embeddings: Embeddings,
        vector_store_type: str,
        vector_store_name: str,
        documents: List[Document]
) -> Union[Chroma, FAISS]:
    """
    This function is used to create or load the vector store.
    If user provided name exists, it will load the vector store.
    If user provided name does not exist, it will create a new vector store and provide uniquehash generated ids to each document.
    """
    if vector_store_type == "chroma":
        if os.path.exists(f"{VECTOR_STORE_PATH}{vector_store_name}"):
            vector_store = Chroma(
                persist_directory=f"{VECTOR_STORE_PATH}{vector_store_name}",
                embedding_function=embeddings
            )
        else:
            vector_store = Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=f"{VECTOR_STORE_PATH}{vector_store_name}",
                ids=[_get_string_hash(doc.page_content) for doc in documents]
            )
    elif vector_store_type == "faiss":
        if os.path.exists(f"{VECTOR_STORE_PATH}{vector_store_name}"):
            vector_store = FAISS.load_local(
                folder_path=f"{VECTOR_STORE_PATH}{vector_store_name}",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vector_store = FAISS.from_documents(
                documents,
                embeddings,
                ids=[_get_string_hash(doc.page_content) for doc in documents]
            )
            vector_store.save_local(f"{VECTOR_STORE_PATH}{vector_store_name}")
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")
    return vector_store