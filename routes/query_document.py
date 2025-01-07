from fastapi import APIRouter, Query, File, UploadFile
from services.rag import load_document, split_document, embedder_by_nvidia, reranker_by_nvidia, create_or_load_vector_store, llm_by_nvidia
from utils.document_comparison import add_unique_documents
from services.llm_chain import inference_chain_rag_with_reranker
from langchain_core.prompts import ChatPromptTemplate
from constants import RETRIEVE_K, RERANK_TOP_N, NVIDIA_LLM_MODEL

query_document_router = APIRouter()


@query_document_router.post("/query_document")
async def query_document(
    query: str = Query(..., description="Query for the document"),
    file: UploadFile = File(..., description="File to be queried"),
    vector_store_type: str = Query(..., description="Type of vector store"),
    vector_store_name: str = Query(..., description="Name of the vector store"),
    retrieve_k: int = Query(None, description="Number of documents to retrieve initially"),
    rerank_top_n: int = Query(None, description="Number of documents to use after reranking"),
    model: str = Query(None, description="Model to be used"),
):
    documents = load_document(file)
    documents = split_document(documents)
    embeddings = embedder_by_nvidia()
    reranker = reranker_by_nvidia()
    vector_store = create_or_load_vector_store(
        documents=documents, 
        embeddings=embeddings,
        vector_store_type=vector_store_type,
        vector_store_name=vector_store_name
    )
    add_unique_documents(
        documents=documents, 
        vector_store=vector_store,
        vector_store_type=vector_store_type,
        vector_store_name=vector_store_name
    )
    template = [
        ("system", "You are a helpful assistant that answers concisely. You are given the following context: {context}."),
        ("human", "{input}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages=template)
    if retrieve_k is None:
        retrieve_k = RETRIEVE_K
    if rerank_top_n is None:
        rerank_top_n = RERANK_TOP_N
    if model is None:
        model = NVIDIA_LLM_MODEL
    llm = llm_by_nvidia(model=model)
    chain = inference_chain_rag_with_reranker(
        vectorstorage=vector_store, 
        llm=llm,
        prompt_template=prompt_template,
        reranker=reranker,
        retrieve_k=retrieve_k,
        rerank_top_n=rerank_top_n
    )
    response = chain.invoke({"input": query})
    return {"chain_response": response["answer"]}

