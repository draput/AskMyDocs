""" Streamlit UI """


import os
import re
from pathlib import Path

import streamlit as st

import help_texts
import langchain_helper as lch


# __ LIB ______________________________________________________________________


@st.cache_data()
def get_lm_components(model_name: str, openai_api_key: str) -> tuple[lch.BaseLanguageModel, lch.OpenAIEmbeddings]:
    return lch.get_lm_components(model_name, openai_api_key)


@st.cache_data()
def load_documents_from_files(
    files: list[lch.FileIO],
    loader_type: str,
    loader_mode: str,
    **kwargs: object,
) -> list[lch.Document]:
    return lch.load_documents_from_files(files, loader_type, loader_mode, **kwargs)


@st.cache_data()
def load_and_split_documents(
    files: list[lch.FileIO],
    loader_type: str,
    loader_mode: str,
    splitter_type: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    **kwargs: object,
) -> tuple[list[lch.Document], list[lch.Document]]:
    documents = load_documents_from_files(
        files,
        loader_type,
        loader_mode,
        **kwargs,
    )
    document_chunks = lch.split_documents(
        documents,
        splitter_type,
        chunk_size,
        chunk_overlap,
    )

    return documents, document_chunks


# def load_documents(
#     documents: list[lch.FileIO],
#     _embeddings_engine: lch.OpenAIEmbeddings,  # undesrcore means it won't be hashed for caching
#     chunk_size: int = 2000,
#     chunk_overlap: int = 200,
# ) -> tuple[lch.FAISS, list[lch.Document]]:
#     st.balloons()
#     return lch.load_documents(documents, _embeddings_engine, chunk_size, chunk_overlap)


def get_available_stores() -> list[str]:
    return [entry.name for entry in Path("db").glob("*/") if entry.is_dir()]


# __ VIEW _____________________________________________________________________


st.set_page_config("Ask my docs", "🤖", "wide")
col1, col2 = st.columns([7, 3])

if "available_stores" not in st.session_state:
    st.session_state.available_stores = get_available_stores()

with st.sidebar as sb:
    with st.expander("Language Model"):
        selected_model = st.selectbox(
            "Language Model:",
            lch.supported_models,
            4,
            key="language_model",
            help=help_texts.select_model_help_text,
        )

        if os.getenv("OPENAI_API_KEY"):
            st.session_state.api_key = os.getenv("OPENAI_API_KEY")

        if "api_key" not in st.session_state:
            api_key = st.text_input(
                "OpenAI API Key:",
                "",
                key="api_key_input",
                type="password",
                help=help_texts.openai_api_key_help_text,
            )
            if api_key:
                st.session_state.api_key = api_key
                st.experimental_rerun()
            st.stop()
        st.selectbox(
            "Embeddings Model:",
            ["OpenAI", "Prem", "..."],
            0,
            key="embeddings_model",
            help="...",
        )

    with st.expander("Prepare Documents"):
        st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            key="uploaded_files",
            help="...",
        )

        st.selectbox(
            "Loader type:",
            ["auto", "pypdf", "..."],
            0,
            key="loader_type",
            help="...",
        )

        st.selectbox(
            "Loader mode:",
            ["single", "elements", "paged"],
            0,
            key="loader_mode",
            help="...",
        )

        if st.session_state.uploaded_files:
            st.selectbox(
                "Splitter type:",
                ["Recursive character splitter", "Character splitter", "...", None],
                0,
                key="splitter_type",
                help="...",
            )
            # TODO: if not None
            st.number_input(
                "Document Chunk Size [characters]:",
                500,
                10000,
                1000,
                key="chunk_size",
                help=help_texts.chunk_size_help_text,
            )

            st.number_input(
                "Chunks Overlap Size [characters]:",
                0,
                2000,
                200,
                key="chunk_overlap",
                help=help_texts.chunk_overlap_help_text,
            )

    with st.expander("Data Store"):
        if st.session_state.uploaded_files:
            st.selectbox(
                "Data Store Type:",
                ["FAISS", "REDIS"],
                key="selected_store_type",
                help="...",
            )
            st.text_input(
                "Save Store Name:",
                "my_store",
                key="store_name",
                help=help_texts.index_name_help_text,
            )
            # TODO: add or merge, depending if store already exists
            st.button(
                "Save Store",
                key="save_store",
            )
        else:
            st.selectbox(
                "Select Local Store:",
                st.session_state.available_stores,
                key="selected_store",
                help=help_texts.selected_search_index_help_text,
            )
            st.button(
                "Load Store",
                key="load_store",
            )

if "vector_store" in st.session_state:
    with st.sidebar:
        with st.expander("Q&A Settings"):
            st.selectbox(
                "Local Store Search Mode:",
                ["similarity", "similarity_score_threshold", "max_marginal_relevance"],
                1,
                key="search_type",
                help=help_texts.search_type_help_text,
            )

            match st.session_state.search_type:
                case "similarity":
                    st.session_state.similarity_threshold = 0.0  # not used
                case "similarity_score_threshold":
                    st.number_input(
                        "Similarity Threshold:",
                        0.0,
                        1.0,
                        0.67,
                        0.01,
                        key="similarity_threshold",
                        help=help_texts.similarity_threshold_help_text,
                    )
                case "max_marginal_relevance":
                    st.number_input(
                        "Diversity:",
                        0.0,
                        1.0,
                        0.5,
                        0.01,
                        key="diversity",
                        help=help_texts.diversity_help_text,
                    )

            st.number_input(
                "Maximum number of relevant document chunks:",
                1,
                250,
                15,
                key="max_relevant_doc_chunks",
                help=help_texts.max_relevant_doc_chunks_help_text,
            )

            st.selectbox(
                "LM Interaction Strategy:",
                ["stuff", "map_reduce", "refine", "map_rerank"],
                0,
                key="chain_type",
                help=help_texts.chain_type_help_text,
            )

# __ CONTROLLER _______________________________________________________________

# st.json(st.session_state)

try:
    language_model, embeddings_model = get_lm_components(
        str(selected_model),
        str(st.session_state.api_key),
    )

    if st.session_state.uploaded_files:
        st.session_state["documents"], st.session_state["document_chunks"] = load_and_split_documents(
            st.session_state.uploaded_files,
            st.session_state.loader_type,
            st.session_state.loader_mode,
            st.session_state.splitter_type,
            st.session_state.chunk_size,
            st.session_state.chunk_overlap,
        )
        # st.json(st.session_state)

        with col1:
            no_files = len(st.session_state.uploaded_files)
            no_documents = len(st.session_state.documents)
            no_chunks = len(st.session_state.document_chunks)
            no_characters, no_tokens = lch.get_documents_statistics(st.session_state.document_chunks, language_model)
            st.caption(
                f"{no_files} files, {no_documents} documents,  {no_chunks} chunks, {no_tokens} tokens, {no_characters} characters<br>",
                unsafe_allow_html=True,
            )

    if "save_store" in st.session_state and st.session_state.save_store:
        st.balloons()
        vector_store = lch.load_store(st.session_state.document_chunks, embeddings_model)
        # TODO: load or merge
        vector_store.save_local(str(Path("db") / st.session_state.store_name))
        st.session_state["vector_store"] = vector_store
        with col1:
            st.caption(f'A local store was initialized and saved under the name "{st.session_state.store_name}"')
        st.experimental_rerun()

    if "load_store" in st.session_state and st.session_state.load_store:
        try:
            vector_store = lch.FAISS.load_local(f"db/{st.session_state.selected_store}", embeddings_model)
            st.session_state["vector_store"] = vector_store
            with col1:
                st.subheader(f'"The {st.session_state.selected_store}" store was selected')
            st.experimental_rerun()
        except RuntimeError:
            st.warning("No local store selected")

    if "vector_store" in st.session_state:
        with col1:
            input_prompt = st.text_area("Enter your question here:", key="input_prompt")

        if input_prompt:
            relevant_documents_chunks = lch.search_vector_store(
                input_prompt,
                st.session_state.vector_store,
                st.session_state.search_type,
                st.session_state.get("similarity_threshold", 0.0),
                st.session_state.get("diversity", 0.0),
                st.session_state.max_relevant_doc_chunks,
            )
            with col1:
                no_characters, no_tokens = lch.get_documents_statistics(relevant_documents_chunks, language_model)
                st.caption(
                    f"{len(relevant_documents_chunks)} relevant document chunks found with a total of "
                    f"{no_tokens} tokens and {no_characters} characters"
                )
                st.balloons()

            with col2:
                for i, chunk in enumerate(relevant_documents_chunks):
                    with st.expander(f"Document Chunk #{i + 1}"):
                        st.markdown(chunk.metadata)
                        text = re.sub(r"\s+", " ", chunk.page_content)  # FIXME - rise level of abstraction
                        st.markdown(f'<p style="font-size:12px;"> {text} </p><hr>', unsafe_allow_html=True)

            reply = lch.ask_question(
                input_prompt, relevant_documents_chunks, language_model, st.session_state.chain_type
            )
            with col1:
                style = "" if len(relevant_documents_chunks) > 0 else 'style="color:red;"'
                st.write(f"<p {style}><i> {reply} </i></p>", unsafe_allow_html=True)


except lch.InvalidRequestError as e:
    st.error(e.user_message)
