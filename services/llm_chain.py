from langchain_core.runnables.base import Runnable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_nvidia_ai_endpoints import NVIDIARerank, ChatNVIDIA
from langchain_core.documents import Document
from langchain.schema.runnable import RunnableLambda
from typing import List

def inference_chain_rag_with_reranker(
        llm: ChatNVIDIA,
        vectorstorage: VectorStore,
        prompt_template: PromptTemplate,
        reranker: NVIDIARerank,
        retrieve_k: int,
        rerank_top_n: int
    ) -> Runnable:
    """
    This function integrates a re-ranker into the Retrieval-QA chain.
    Args:
        llm: nvidia llm
        vectorstorage: vector store
        prompt_template: prompt template
        reranker: reranker from langchain_nvidia_ai_endpoints
        retrieve_k: number of initial documents to retrieve from vector store
        rerank_top_n: number of top-ranked documents to use after re-ranking
    """
    # Create a retriever from the vector store
    retriever = vectorstorage.as_retriever(search_kwargs={'k': retrieve_k})
    
    def re_rank_documents(query: str, documents: List[Document]) -> List[Document]:
        """Re-rank documents using NVIDIA Reranker."""
        response = reranker.compress_documents(query=query, documents=documents)
        return response[:rerank_top_n]

    def re_ranked_retriever(inputs: dict) -> List[Document]:
        """Retrieve documents and apply re-ranking for RunnableLambda."""
        query = inputs["input"]  # Extract query from input dictionary
        initial_docs = retriever.invoke(query)
        re_ranked_docs = re_rank_documents(query, initial_docs)
        return re_ranked_docs
    
    # Create the QA chain
    qa_chain = create_stuff_documents_chain(
        llm=llm, 
        prompt=prompt_template,
    )
    
    # Create a runnable for the re-ranked retriever
    re_ranked_retriever_runnable = RunnableLambda(func=re_ranked_retriever)

    # Combine retriever with QA chain
    retrieval_chain = create_retrieval_chain(re_ranked_retriever_runnable, qa_chain)
    return retrieval_chain