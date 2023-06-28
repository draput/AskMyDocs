"""Help Texts for the Streamlit Widgets"""

openai_api_key_help_text = "Generate your OpenAI API key [here](https://platform.openai.com/account/api-keys)"

select_model_help_text = """
        Select the language model based on its [performance](https://platform.openai.com/docs/models) and [price](https://openai.com/pricing)\n
        You can check your usage [here](https://platform.openai.com/account/usage)"""

chunk_size_help_text = "Maximal number of characters pe chunk"
chunk_overlap_help_text = "Charaters ovelaps between chunks"
index_name_help_text = "The name under which a store will be saved locally"
selected_search_index_help_text = "Load an existing store with this name"

search_type_help_text = """
        How to search the local store for the relevant document cunks\n
        similarity - returns document chunks similar to the query\n
        similarity_score_treshold - returns documents chunks similar to the query if they are above a given similarity treshold\n
        max_marginal_relevance - returns document chunks optimized for similarity AND diversity, considering a similarity treshold too"""

similarity_threshold_help_text = """
        Consider only the document chunks above this relevance score\n
        0 - dissimilar\n
        1 - most similar
"""

diversity_help_text = """
        Determines the degree of diversity among the results\n
        0 - maximum diversity\n
        1 - minimum diversity
"""

max_relevant_doc_chunks_help_text = "Take not more than this number of document chunks, when interacting with the ALM"

chain_type_help_text = """
        How to interact with the langueage model:\n
        stuff - Stuff everyting in one prompt. Only one API call, but it can get larger as the token limit.\n
        map_reduce - Summarize each chunk, answer from the summary. Better for larger documents, makes more (paralellizable) API calls.\n
        refine - Ask the first chunk, than add next chunk to the answer, repeat. Can pull more relevant context, also many (sequential) API calls. \n
        map_rerank - Ask each chunk for an aswer with a score, and pick up the one with the best score. Cannot combine information from different chunks.
        """
