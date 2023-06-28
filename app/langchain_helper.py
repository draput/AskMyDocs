"""Convenience functions on the top of LangChain"""

import os

from io import FileIO
from pathlib import Path
from typing import Literal

from langchain.base_language import BaseLanguageModel
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    # PyPDFLoader, PDFMinerLoader, PDFMinerPDFasHTMLLoader, PDFPlumberLoader,
    # UnstructuredPDFLoader,
    UnstructuredFileLoader,
    UnstructuredFileIOLoader,
)
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore, FAISS

from openai.error import InvalidRequestError
from langchain.schema import Document

import tiktoken

# from langchain.schema import BaseRetriever


# See https://openai.com/pricing


supported_chat_models = [
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
]

supported_llm_models = [
    "text-davinci-003",
    "text-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
]

supported_models = supported_chat_models + supported_llm_models


def get_lm_components(
    model_name: str,
    openai_api_key: str,
) -> tuple[BaseLanguageModel, OpenAIEmbeddings]:
    match model_name:
        case model_name if model_name in supported_llm_models:
            language_model = OpenAI(
                model=model_name,
                client="",
                temperature=0.0,
                batch_size=8,
                openai_api_key=openai_api_key,
                verbose=True,
            )
        case model_name if model_name in supported_chat_models:
            language_model = ChatOpenAI(
                client="",
                model=model_name,
                temperature=0.0,
                openai_api_key=openai_api_key,
                verbose=True,
            )
        # case _:
        #     language_model = ...

    # NOTE: always use the text-embedding-ada-002 embedings model
    embeddings_engine = OpenAIEmbeddings(client="", model="text-embedding-ada-002", openai_api_key=openai_api_key)

    return language_model, embeddings_engine


def load_documents(
    document_files: list[FileIO] | list[str] | list[Path],
    embeddings_engine: OpenAIEmbeddings,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> tuple[FAISS, list[Document]]:
    docs = []
    for file in document_files:
        match file:
            case str() | Path():
                loader = UnstructuredFileLoader(str(file))
            case _:  # FIXME: be explicit for FileIO!
                loader = UnstructuredFileIOLoader(file)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document_chunks = text_splitter.split_documents(docs)
    # print(f"{len(docs)} docs, {len(doc_chunks)} doc_chunks")

    print("$$$")
    vector_store = FAISS.from_documents(document_chunks, embeddings_engine)  # API CALL$

    return vector_store, document_chunks


def search_vector_store(
    query: str,
    vector_store: VectorStore,
    search_type: str,  # Literal["similarity", "similarity_score_threshold", "max_marginal_relevance"] = "similarity",
    relevance_score_threshold: float = 0.6,  # 0 - dissimmilar, 1 - most similar
    diversity: float = 0.5,  # 0 - max diversity, 1 - min diversity
    max_returned_document_chunks: int = 15,
) -> list[Document]:
    match search_type:
        case "similarity":
            relevant_document_chunks = vector_store.similarity_search(
                query,
                k=max_returned_document_chunks,
            )
        case "similarity_score_threshold":
            relevant_doc_chunks_with_score = vector_store.similarity_search_with_relevance_scores(
                query, k=max_returned_document_chunks
            )
            relevant_document_chunks = [
                ds[0] for ds in relevant_doc_chunks_with_score if ds[1] > relevance_score_threshold
            ]

        case "max_marginal_relevance":
            relevant_document_chunks = vector_store.max_marginal_relevance_search(
                query, k=max_returned_document_chunks, lambda_mult=diversity, fetch_k=max_returned_document_chunks
            )
        case _:
            relevant_document_chunks = []

    return relevant_document_chunks


def ask_question(
    query: str,
    relevant_document_chunks: list[Document],
    language_model: BaseLanguageModel,
    chain_type: str | Literal["stuff", "refine", "map_reduce", "map_rerank"] = "stuff",
) -> str:
    qa_chain = load_qa_chain(language_model, chain_type=chain_type, verbose=True)
    response = qa_chain.run(input_documents=relevant_document_chunks, question=query)

    return response


# if __name__ == "__main__":
#     file_path = Path("docs") / "VG_RAW/FileTab_Preferences_General.txt"
#     store_name = "home_tab_1000_200"

#     openai_api_key = os.environ.get("OPENAI_API_KEY", "")
#     language_model, embeddings_engine = get_lm_components("gpt-3.5-turbo", openai_api_key)

#     # vector_store, doc_chunks = load_documents([file_path], embeddings_engine, 1000, 200)
#     # print(f"Loaded into {len(doc_chunks)} chunks ")
#     # vector_store.save_local(f"db/{store_name}")

#     vector_store = FAISS.load_local(f"db/{store_name}", embeddings_engine)
#     query = "8 bit"

#     relevant_documents_chunks = search_vector_store(
#         query,
#         vector_store,
#         "similarity_score_threshold",
#         relevance_score_threshold=0.6,
#         max_returned_document_chunks=50,
#     )
#     print(f"{len(relevant_documents_chunks)} chunks found")
# #     reply = ask_question(query, relevant_documents_chunks, language_model)

# #     print(reply)
