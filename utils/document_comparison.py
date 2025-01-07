import hashlib
from typing import List
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from constants import VECTOR_STORE_PATH

def _get_string_hash(input_string: str) -> str:
    sha256_hash = hashlib.sha256()
    # Convert the string to bytes and update the hash
    sha256_hash.update(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()

def add_unique_documents(documents: List[Document], vector_store: VectorStore, vector_store_type: str, vector_store_name: str):
    if vector_store_type == "chroma":
        doc_ids = [_get_string_hash(doc.page_content) for doc in documents]
        documents = [(doc, doc_ids[idx]) for idx, doc in enumerate(documents) if len(vector_store.get([doc_ids[idx]])['ids']) == 0]
        if len(documents) > 0:
            vector_store.add_documents(documents=[doc[0] for doc in documents], ids=[doc[1] for doc in documents])
            print(f"Added {len(documents)} unique documents to the vector store. (chroma)")
        else:
            print("All documents already exist in the vector store. (chroma)")
    elif vector_store_type == "faiss":
        doc_ids = [_get_string_hash(doc.page_content) for doc in documents]
        documents = [(doc, doc_ids[idx]) for idx, doc in enumerate(documents) if doc_ids[idx] not in vector_store.index_to_docstore_id.values()]
        if len(documents) > 0:
            vector_store.add_documents(documents=[doc[0] for doc in documents], ids=[doc[1] for doc in documents])
            vector_store.save_local(f"{VECTOR_STORE_PATH}{vector_store_name}")
            print(f"Added {len(documents)} unique documents to the vector store. (faiss)")
        else:
            print("All documents already exist in the vector store. (faiss)")