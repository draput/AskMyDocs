"""Convenience functions on the top of LangChain"""

import os

import io
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
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain

from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.qa_generation import prompt
from langchain.chains.query_constructor import parser, prompt, schema, ir

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
                batch_size=15,
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
    embeddings_engine = OpenAIEmbeddings(
        client="", model="text-embedding-ada-002", openai_api_key=openai_api_key
    )

    return language_model, embeddings_engine


def load_documents_from_files(
    document_files: list[FileIO] | list[str] | list[Path],
    loder_type: str = "unstructured",
    mode: str = "single",  # "single", "elements", "paged"
    **kwargs: object,
) -> list[Document]:
    print(document_files)
    all_documents = []
    for file in document_files:
        # if (
        #     isinstance(file, io.FileIO) or type(file).__name__ == "UploadedFile"
        # ):  # HACK: for streamlit uploaded files
        #     loader = UnstructuredFileIOLoader(file, mode)
        #     documents_for_file = loader.load()
        #     # HACK: add metadata manually for file IO
        #     for document in documents_for_file:
        #         document.metadata["source"] = file.name
        # elif isinstance(file, str) or isinstance(file, Path):
        #     loader = UnstructuredFileLoader(str(file), mode)
        #     documents_for_file = loader.load()
        # else:
        #     documents_for_file = []

        loader = UnstructuredFileIOLoader(file, mode, strategy="fast")
        documents_for_file = loader.load()
        all_documents.extend(documents_for_file)

    return all_documents


def split_documents(
    documents: list[Document],
    splitter_type: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    document_chunks = text_splitter.split_documents(documents)

    return document_chunks


def load_store(
    document_chunks: list[Document],
    embeddings_model: OpenAIEmbeddings,
    store_type: str = "FAISS",
) -> VectorStore:
    vector_store = FAISS.from_documents(document_chunks, embeddings_model)  # API CALL$
    return vector_store

    # print(f"{len(docs)} docs, {len(doc_chunks)} doc_chunks")

    # print("$$$")
    # vector_store = FAISS.from_documents(document_chunks, embeddings_engine)  # API CALL$

    ##return vector_store, document_chunks


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
                include_metadata=True,
            )
        case "similarity_score_threshold":
            relevant_doc_chunks_with_score = (
                vector_store.similarity_search_with_relevance_scores(
                    query,
                    k=max_returned_document_chunks,
                )
            )
            relevant_document_chunks = [
                ds[0]
                for ds in relevant_doc_chunks_with_score
                if ds[1] > relevance_score_threshold
            ]

        case "max_marginal_relevance":
            relevant_document_chunks = vector_store.max_marginal_relevance_search(
                query,
                k=max_returned_document_chunks,
                lambda_mult=diversity,
                fetch_k=max_returned_document_chunks,
            )
        case _:
            relevant_document_chunks = []

    return relevant_document_chunks


def get_documents_statistics(
    documents: list[Document], language_model: BaseLanguageModel
) -> tuple[int, int]:
    no_chars = 0
    no_tokens = 0
    for document in documents:
        text = document.page_content
        no_chars += len(text)
        no_tokens += language_model.get_num_tokens(text)
    return no_chars, no_tokens


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
#     file_path = Path("docs") / "xxx.txt"
#     # store_name = "home_tab_1000_200"
#     store_name = "xxx"

#     openai_api_key = os.environ.get("OPENAI_API_KEY", "")
#     language_model, embeddings_model = get_lm_components(
#         "gpt-3.5-turbo", openai_api_key
#     )

#     # file_io = FileIO(str(file_path))
#     # docs = load_documents_from_files([file_io], mode="single")
#     docs = load_documents_from_files([file_path], "", mode="single")

#     document_chunks = split_documents(
#         docs,
#         "",
#         1000,
#         200,
#     )

#     vector_store = load_store(
#         document_chunks,
#         embeddings_model,
#     )
#     vector_store.save_local(f"db/{store_name}")  # type: ignore

#     print()

#     # vector_store, doc_chunks = load_documents([file_path], embeddings_engine, 1000, 200)
#     # retriever = vector_store.as_retriever(
#     #     search_type="similarity"
#     # )  # allowed_search_types
#     # print(f"Loaded into {len(doc_chunks)} chunks ")
#     # vector_store.save_local(f"db/{store_name}")

#     vector_store = FAISS.load_local(f"db/{store_name}", embeddings_model)

#     query = "How can I write is better code? Use only the povided context."
#     relevant_doc_chunks = vector_store.similarity_search(query=query)  # $$$

#     print()

#     # qa = load_qa_chain(
#     #     llm=language_model,
#     #     chain_type="stuff",
#     #     verbose=True,
#     # )

#     # rsp = qa(
#     #     inputs={"question": query, qa.input_key: relevant_doc_chunks},
#     #     return_only_outputs=False,
#     #     include_run_info=False,
#     # )
#     # print(rsp)

#     # for doc_chunk in relevant_doc_chunks:
#     #     doc_chunk.metadata["source"] = "some_source"

#     # qa = load_qa_with_sources_chain(
#     #     llm=language_model,
#     #     chain_type="stuff",
#     #     verbose=True,
#     # )

#     # # reply = qa.run(input_documents=relevant_doc_chunks, question=query)
#     # # qa is a callable
#     # rsp = qa(
#     #     inputs={"question": query, "input_documents": relevant_doc_chunks},
#     #     return_only_outputs=False,
#     #     include_run_info=True,
#     # )
#     # print(rsp)

#     relevant_documents_chunks = search_vector_store(
#         query,
#         vector_store,
#         "similarity_score_threshold",
#         relevance_score_threshold=0.6,
#         max_returned_document_chunks=10,
#     )
#     print(f"{len(relevant_documents_chunks)} chunks found")

#     reply = ask_question(query, relevant_documents_chunks, language_model)

#     print(reply)
