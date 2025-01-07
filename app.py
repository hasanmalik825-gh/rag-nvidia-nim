import streamlit as st
from services.rag import create_or_load_vector_store, load_document, split_document, embedder_by_nvidia, reranker_by_nvidia, llm_by_nvidia
from utils.document_comparison import add_unique_documents
from routes.query_document import inference_chain_rag_with_reranker
from langchain_core.prompts import ChatPromptTemplate
from constants import RETRIEVE_K, RERANK_TOP_N, NVIDIA_LLM_MODEL

st.title("Chat with Documents using NVIDIA")
st.sidebar.title("Settings")

nvidia_api_key = st.sidebar.text_input("NVIDIA API Key", type="password")
vector_store_type = st.sidebar.selectbox("Vector store type", ["chroma", "faiss"])
vector_store_name = st.sidebar.text_input("Vector store name")
model_id = st.sidebar.text_input("NVIDIA LLM model id", value=NVIDIA_LLM_MODEL)
retrieve_k = st.sidebar.number_input("Retrieve k", value=RETRIEVE_K)
rerank_top_n = st.sidebar.number_input("Rerank top n", value=RERANK_TOP_N)

template = [
        ("system", "You are a helpful assistant that answers concisely. You are given the following context: {context}."),
        ("human", "{input}"),
    ]
prompt_template = ChatPromptTemplate.from_messages(messages=template)

file = st.file_uploader("Upload file", type=["pdf", "txt"])
query = st.text_input("Enter query")

if st.button("Chat"):
    if file and query and vector_store_type and vector_store_name and retrieve_k and rerank_top_n and model_id and nvidia_api_key:
        llm = llm_by_nvidia(model=model_id, api_key=nvidia_api_key)
        embeddings = embedder_by_nvidia(api_key=nvidia_api_key)
        reranker = reranker_by_nvidia(api_key=nvidia_api_key)

        documents = load_document(file)
        documents = split_document(documents)
        vector_store = create_or_load_vector_store(
            embeddings=embeddings,
            vector_store_type=vector_store_type,
            vector_store_name=vector_store_name,
            documents=documents
        )
        add_unique_documents(
            documents=documents, 
            vector_store=vector_store,
            vector_store_type=vector_store_type,
            vector_store_name=vector_store_name
        )
        st.success("Documents added to vector store")
        st.info("Now thinking on query...")
        chain = inference_chain_rag_with_reranker(
        vectorstorage=vector_store, 
        llm=llm,
        prompt_template=prompt_template,
        reranker=reranker,
        retrieve_k=retrieve_k,
        rerank_top_n=rerank_top_n
    )
        response = chain.invoke({"input": query})
        st.success("Got the answer, Yea!")
        st.write(response["answer"])
    else:
        st.error("Please fill all the fields")


