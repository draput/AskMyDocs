""" Streamlit UI """

import os
from pathlib import Path

import streamlit as st

import help_texts
import langchain_helper as lch


# __ LIB ______________________________________________________________________


@st.cache_data()
def get_lm_components(model_name: str, openai_api_key: str) -> tuple[lch.BaseLanguageModel, lch.OpenAIEmbeddings]:
    return lch.get_lm_components(model_name, openai_api_key)


@st.cache_data(persist=True)
def load_documents(
    documents: list[lch.FileIO],
    _embeddings_engine: lch.OpenAIEmbeddings,  # undesrcore means it won't be hashed for caching
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> tuple[lch.FAISS, list[lch.Document]]:
    st.balloons()
    return lch.load_documents(documents, _embeddings_engine, chunk_size, chunk_overlap)


def get_available_stores() -> list[str]:
    return [entry.name for entry in Path("db").glob("*/") if entry.is_dir()]


# __ VIEW _____________________________________________________________________


st.set_page_config("Ask my docs", "🤖")

if "available_stores" not in st.session_state:
    st.session_state.available_stores = get_available_stores()

with st.sidebar as sb:
    selected_model = st.selectbox("Language Model:", lch.supported_models, 4, help=help_texts.select_model_help_text)

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

    with st.expander("Prepare document(s)"):
        st.file_uploader("Upload document(s)", key="uploaded_files", accept_multiple_files=True)

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
        st.text_input("Local Store Name:", "my_store", key="store_name", help=help_texts.index_name_help_text)
        st.button("Save Store", key="save_search_store")

    with st.expander("Q&A Settings"):
        st.selectbox(
            "Select Local Store:",
            st.session_state.available_stores,
            key="selected_store",
            help=help_texts.selected_search_index_help_text,
        )

        st.selectbox(
            "Local Store Search Mode:",
            ["similarity", "similarity_score_threshold", "max_marginal_relevance"],
            0,
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
                    0.5,
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
            "Maximum number of relevant search results:",
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
    vector_store = None
    language_model, embeddings_engine = get_lm_components(str(selected_model), str(st.session_state.api_key))

    if st.session_state.uploaded_files:
        vector_store, document_chunks = load_documents(
            st.session_state.uploaded_files,
            embeddings_engine,
            st.session_state.chunk_size,
            st.session_state.chunk_overlap,
        )
        if vector_store:
            st.caption(
                "A temporary local store was created from the uploaded documents "
                f"from the resulting {len(document_chunks)} chunks"
            )

        if st.session_state.save_search_store:
            vector_store.save_local(str(Path("db") / st.session_state.store_name))
            st.session_state.available_stores = get_available_stores()
            st.success(f'The temporary local store was saved under the "{st.session_state.store_name}" name')
            st.experimental_rerun()

    if not vector_store:
        try:
            vector_store = lch.FAISS.load_local(f"db/{st.session_state.selected_store}", embeddings_engine)
            st.subheader(f'"{st.session_state.selected_store}" was selected')
        except RuntimeError:
            st.warning("No local store selected")

    if vector_store:
        input_prompt = st.text_area("Enter your question here:", key="input_prompt")

        if input_prompt:
            relevant_documents_chunks = lch.search_vector_store(
                input_prompt,
                vector_store,
                st.session_state.search_type,
                st.session_state.get("similarity_threshold", 0.0),
                st.session_state.get("diversity", 0.0),
                st.session_state.max_relevant_doc_chunks,
            )
            st.caption(f"{len(relevant_documents_chunks)} relevant document chunks found")
            st.balloons()
            reply = lch.ask_question(
                input_prompt, relevant_documents_chunks, language_model, st.session_state.chain_type
            )
            st.write(f"*{reply}*", unsafe_allow_html=True)

except lch.InvalidRequestError as e:
    st.error(e.user_message)
