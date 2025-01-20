import argparse
import os
import re
import openai
import logging
import time
import numpy as np
from collections import deque
import json
import pandas as pd
from tqdm import tqdm
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
import faiss
from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from transformers import GPT2TokenizerFast
from langchain_ollama import ChatOllama



# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY_CIMPLE")

# OpenAI Engine, feel free to change
ENGINE = 'gpt-4o-mini'
#EMBEDDINGS_MODEL = "text-embedding-3-large"
EMBEDDINGS_MODEL = "text-embedding-3-small"
# Max number of LLM retries
MAX_GPT_CALLS = 5

if ENGINE == 'gpt-4o-mini':
    max_tokens_min = 10000000
else:
    max_tokens_min = 2000000

TOKEN_THRESHOLD = int(0.75 * max_tokens_min)
INTERVAL_SECONDS = 45  # Time interval to monitor
DELAY_SECONDS = 60 #Delay if token_threshold is reached


usage_log = deque()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("file_management.log"), logging.StreamHandler()]
)

def count_tokens(text):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    return len(tokens)

def semantic_similarity_search(scrapped_text, args, max_prompt_tokens = 4000, prompt_params=None, numb_similar_docs=20):
    temp_file_path  = args.input_path.split('.jsonl')[0] + '.txt'
    with open(temp_file_path, 'w', encoding="utf-8") as text_file:
        text_file.write(scrapped_text)
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    raw_documents = TextLoader(temp_file_path, encoding='UTF-8').load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(max_prompt_tokens/0.75), chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    db = Chroma.from_documents(documents, embeddings)
    embedding_vector = embeddings.embed_query(prompt_params['decomposed_justification'])
    #Return the top 20 most similar documents
    docs = db.similarity_search_by_vector(embedding_vector, k=numb_similar_docs)
    response = ''
    for doc in docs:
        response = response + doc.page_content
    os.remove(temp_file_path)
    return response

def calculate_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def faiss_similarity_index(scrapped_text, statement_to_compare, max_prompt_tokens=4000, token_limit=8191):

    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)  
    
    # Split the scrapped text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(max_prompt_tokens / 0.75), chunk_overlap=0)
    scrapped_chunks = text_splitter.create_documents([scrapped_text])
    
    # Embed the chunks of scrapped_text
    scrapped_embeddings = [embeddings.embed_query(doc.page_content) for doc in scrapped_chunks]
    
    # Check if statement_to_compare exceeds token limit
    if count_tokens(statement_to_compare) > token_limit:
        # Split statement_to_compare into chunks
        statement_chunks = text_splitter.create_documents([statement_to_compare])
    else:
        # Treat the whole statement as a single chunk
        statement_chunks = [statement_to_compare]
    
    # Embed each chunk of statement_to_compare and calculate similarities
    similarity_scores = []
    for statement_chunk in statement_chunks:
        statement_embedding = embeddings.embed_query(statement_chunk.page_content if hasattr(statement_chunk, "page_content") else statement_chunk)
        
        # Calculate similarity with each scrapped chunk and take the maximum for this statement chunk
        chunk_similarities = [
            calculate_similarity(statement_embedding, scrapped_embedding)
            for scrapped_embedding in scrapped_embeddings
        ]
        similarity_scores.append(max(chunk_similarities))
    
    # Compute the average similarity score across all statement chunks
    similarity_index = sum(similarity_scores) / len(similarity_scores)
    return similarity_index

def faiss_similarity_index_old(
    scrapped_text, 
    statement_to_compare, 
    max_prompt_tokens=4000, 
    token_limit=8191  #match `text-embedding-3-small` limit
):
    from tiktoken import get_encoding

    # Initialize tokenizer for token counting
    tokenizer = get_encoding("cl100k_base")

    def count_tokens(text):
        """Count tokens in the given text using the tokenizer."""
        return len(tokenizer.encode(text))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")   
    
    # Split the scrapped text into manageable chunks within token limit
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=token_limit - 10,  # Slightly below token limit for safety
        chunk_overlap=0
    )
    scrapped_chunks = text_splitter.create_documents([scrapped_text])

    # Embed the chunks of scrapped_text
    scrapped_embeddings = [
        embeddings.embed_query(doc.page_content) for doc in scrapped_chunks
    ]
    
    # Handle statement_to_compare token count
    if count_tokens(statement_to_compare) > token_limit:
        # Split statement_to_compare into chunks
        statement_chunks = text_splitter.create_documents([statement_to_compare])
    else:
        # Treat the whole statement as a single chunk
        statement_chunks = [statement_to_compare]

    # Embed each chunk of statement_to_compare and calculate similarities
    similarity_scores = []
    for statement_chunk in statement_chunks:
        # Extract content for embedding
        chunk_content = (
            statement_chunk.page_content
            if hasattr(statement_chunk, "page_content")
            else statement_chunk
        )

        # Check token count and truncate if necessary
        if count_tokens(chunk_content) > token_limit:
            chunk_content = tokenizer.decode(tokenizer.encode(chunk_content)[:token_limit])
        
        # Embed the chunk and compute similarities
        statement_embedding = embeddings.embed_query(chunk_content)
        chunk_similarities = [
            calculate_similarity(statement_embedding, scrapped_embedding)
            for scrapped_embedding in scrapped_embeddings
        ]
        similarity_scores.append(max(chunk_similarities))

    # Compute the average similarity score across all statement chunks
    similarity_index = sum(similarity_scores) / len(similarity_scores)
    return similarity_index


def Faiss_similarity_search(scrapped_text, statement_to_compare, args, max_prompt_tokens = 4000, prompt_params=None, numb_similar_docs=20):
    temp_file_path  = args.input_path.split('.jsonl')[0] + '.txt'
    with open(temp_file_path, 'w', encoding="utf-8") as text_file:
        text_file.write(scrapped_text)
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    raw_documents = TextLoader(temp_file_path, encoding='UTF-8').load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(max_prompt_tokens/0.75), chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    #db = Chroma.from_documents(documents, embeddings)
    #embedding_vector = embeddings.embed_query(prompt_params['decomposed_justification'])
    embedding_vector = embeddings.embed_query(statement_to_compare)
    index = faiss.IndexFlatL2(len(embedding_vector))
    vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    results = vector_store.similarity_search(
    prompt_params['decomposed_justification'],
    k=numb_similar_docs,
)
    
    #docs = db.similarity_search_by_vector(embedding_vector, k=numb_similar_docs)
    response = ''
    for res in results:
        response = response + res.page_content
    return response

def promptLLM(llm, prompt_funcs, scraped_text, start_time, max_prompt_tokens = 4000, prompt_params=None):
    func_names = []
    for prompt_func in prompt_funcs:
        func_names.append(prompt_func.__name__)
    #chunk_size is in words. 1 token approximatelly 0.75 words
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(max_prompt_tokens/0.75), chunk_overlap=500) 
    docs = text_splitter.create_documents([scraped_text])
    num_tokens = llm.get_num_tokens(scraped_text)
    tokens_in_interval = sum(tokens for timestamp, tokens in usage_log)
    
    
    '''if (tokens_in_interval  + num_tokens) > TOKEN_THRESHOLD:
        print(f"Token usage for next summarizatio: {num_tokens} tokens...")
        print(f"Token usage will exceed {TOKEN_THRESHOLD} tokens. Pausing for {DELAY_SECONDS} seconds...")
        time.sleep(DELAY_SECONDS)
        with get_openai_callback() as callback_handler:    
            callback_handler.total_tokens = 0   
        start_time = time.time()
    #If scraped text is smaller than a single prompt chunk, just run the simple prompt
    if num_tokens <= max_prompt_tokens:
        index = func_names.index('construct_prompt')
        prompt = prompt_funcs[index](scraped_text, prompt_params)
        with get_openai_callback() as callback_handler:
            response = llm.invoke(prompt)
    else:
        index = func_names.index('construct_mapreduce_prompt')
        prompt = prompt_funcs[index](prompt_params)
        summary_chain = load_summarize_chain(llm=llm,
                                    chain_type='map_reduce',
                                    map_prompt=prompt['map_prompt'],
                                    combine_prompt=prompt['combine_prompt'],
                                    #verbose=True
                                    )
        with get_openai_callback() as callback_handler:
            response = summary_chain.invoke(docs)
    llm_call_endtime = time.time()
    tokens_used = callback_handler.total_tokens
    # Log the token usage with a timestamp
    usage_log.append((start_time, tokens_used))
    # Remove outdated entries from the log (older than the interval)
    print(f"Time interval in seconds: {(llm_call_endtime - usage_log[0][0])}")
    while usage_log and (time.time() - usage_log[0][0]) > INTERVAL_SECONDS:
        start_time = time.time()
        usage_log.popleft()
        callback_handler.total_tokens = 0
    # Calculate token usage in the last INTERVAL_SECONDS
    tokens_in_interval = sum(tokens for timestamp, tokens in usage_log)
    print(f"Tokens used in summarization: {tokens_used}")
    print(f"Tokens in interval: {tokens_in_interval}")

    return response'''
    if (tokens_in_interval + num_tokens) > TOKEN_THRESHOLD:
        print(f"Token usage for next summarization: {num_tokens} tokens...")
        print(f"Token usage will exceed {TOKEN_THRESHOLD} tokens. Pausing for {DELAY_SECONDS} seconds...")
        time.sleep(DELAY_SECONDS)
        start_time = time.time()

    # If the text fits in a single prompt, use a simple prompt
    if num_tokens <= max_prompt_tokens:
        index = func_names.index('construct_prompt')
        prompt = prompt_funcs[index](scraped_text, prompt_params)

        # Invoke based on the type of LLM
        #if hasattr(llm, "chat"):  # For ChatOllama
        if isinstance(llm, ChatOllama):
            response = llm.invoke(
                input=prompt,
                #model="llama3.2:3b",  # Specify the model
                #temperature=0.7,
                #max_tokens=max_prompt_tokens,
            )
            response_text = response
        else:  # Assume OpenAI LLM
            if hasattr(llm, "callback_manager") and llm.callback_manager is not None:
                with llm.callback_manager as callback_handler:
                    response_text = llm.invoke(prompt)
            else:
                response_text = llm.invoke(prompt)  # For models without a callback_manager

    else:
        # Use Map-Reduce prompt for larger text
        index = func_names.index('construct_mapreduce_prompt')
        prompt = prompt_funcs[index](prompt_params)
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=prompt['map_prompt'],
            combine_prompt=prompt['combine_prompt'],
        )
        if hasattr(llm, "callback_manager") and llm.callback_manager is not None:
            with llm.callback_manager as callback_handler:
                response_text = summary_chain.invoke(docs)
        else:
            response_text = summary_chain.invoke(docs)

    llm_call_endtime = time.time()
    
    # Log token usage
    tokens_used = num_tokens  # Adjust if a more accurate token count is needed
    usage_log.append((start_time, tokens_used))
    
    # Remove outdated entries from the log
    print(f"Time interval in seconds: {(llm_call_endtime - usage_log[0][0])}")
    while usage_log and (time.time() - usage_log[0][0]) > INTERVAL_SECONDS:
        usage_log.popleft()
    
    # Calculate token usage in the last INTERVAL_SECONDS
    tokens_in_interval = sum(tokens for timestamp, tokens in usage_log)
    print(f"Tokens used in summarization: {tokens_used}")
    print(f"Tokens in interval: {tokens_in_interval}")

    return response_text

#The below is merely a playground for experiments as external scripts use this file as a module
def main(args):
    df = pd.read_json(args.input_path, lines=True)
    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end

    #Temperature = 0 as we want the summary to be factual and based on the input text
    llm = ChatOpenAI(temperature = 0, model = ENGINE, api_key = api_key, max_tokens = 1024, max_retries = MAX_GPT_CALLS)

    for i in tqdm(range(start, end)):
        try:
            decomposed_search_hits = df.iloc[i]['decomposed_search_hits']
            for decomposed_search_hit in decomposed_search_hits:
                decomposed_justification = decomposed_search_hit['decomposed_justification']
                decomposed_question = decomposed_search_hit['decomposed_question']
                prompt_params={'decomposed_justification':decomposed_justification, 'decomposed_question':decomposed_question}
                j = 0
                start_time = time.time()
                #Summarize each url page content
                for page_info in decomposed_search_hit['pages_info']:
                    num_tokens = llm.get_num_tokens(page_info['page_content'])
                    #If content is too large, do semantic similarity with justification to return only the relevant chunks
                    if num_tokens >= TOKEN_THRESHOLD:
                        scrapped_text = semantic_similarity_search(llm, page_info['page_content'], args, max_prompt_tokens = 4000, prompt_params=prompt_params)
                        #If semantic similarity search took longer than the set interval for tokens per minute, restart start time
                        if time.time() - start_time > INTERVAL_SECONDS:
                            start_time = time.time()
                    else: 
                        scrapped_text = page_info['page_content']
                    page_info['justification_summary'] = promptLLM(llm, scrapped_text, start_time, prompt_params=prompt_params)         
                    decomposed_search_hit['pages_info'][j] = page_info.copy()
                    j = j + 1
        except Exception as e:
            print("error caught", e)
            print('Dataset row = ', i)
            print('Decomposed Question: ', decomposed_search_hit['decomposed_question'])
            print('Decomposed Justification: ', decomposed_search_hit['decomposed_justification'])
            print('Pages_Info index = ', j)           
            print('Page name: ', page_info['page_name']) 
            print('Page url: ', page_info['page_url'])
            print('Page content length: ', len(page_info['page_content'].strip().split(" ")))

    df.to_json(args.output_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    main(args)
