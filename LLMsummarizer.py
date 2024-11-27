import argparse
import os
import re
import openai
import logging
import time
from collections import deque
import json
import pandas as pd
from tqdm import tqdm
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# OpenAI API Key
#openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI Engine, feel free to change
ENGINE = 'gpt-4o-mini'
# Max number of LLM retries
MAX_GPT_CALLS = 5

if ENGINE == 'gpt-4o-mini':
    max_tokens_min = 200000
else:
    max_tokens_min = 30000

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


def semantic_similarity_search(scrapped_text, args, max_prompt_tokens = 4000, prompt_params={}):
    temp_file_path  = args.input_path.split('.jsonl')[0] + '.txt'
    with open(temp_file_path, 'w', encoding="utf-8") as text_file:
        text_file.write(scrapped_text)
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    raw_documents = TextLoader(temp_file_path, encoding='UTF-8').load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(max_prompt_tokens/0.75), chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    embedding_vector = OpenAIEmbeddings().embed_query(prompt_params['decomposed_justification'])
    #Return the top 20 most similar documents
    docs = db.similarity_search_by_vector(embedding_vector, k=20)
    response = ''
    for doc in docs:
        response = response + doc.page_content
    return response

def summarize_justification(llm, prompt_funcs, scraped_text, start_time, max_prompt_tokens = 4000, prompt_params={}):
    func_names = []
    for prompt_func in prompt_funcs:
        func_names.append(prompt_func.__name__)
    #chunk_size is in words. 1 token approximatelly 0.75 words
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(max_prompt_tokens/0.75), chunk_overlap=500) 
    docs = text_splitter.create_documents([scraped_text])
    num_tokens = llm.get_num_tokens(scraped_text)
    tokens_in_interval = sum(tokens for timestamp, tokens in usage_log)
    if (tokens_in_interval  + num_tokens) > TOKEN_THRESHOLD:
        print(f"Token usage for next summarizatio: {num_tokens} tokens...")
        print(f"Token usage will exceed {TOKEN_THRESHOLD} tokens. Pausing for {DELAY_SECONDS} seconds...")
        time.sleep(DELAY_SECONDS)
        with get_openai_callback() as callback_handler:    
            callback_handler.total_tokens = 0   
        start_time = time.time()
    #If scraped text is smaller than a single prompt chunk, just run the simple prompt
    if num_tokens <= max_prompt_tokens:
        index = func_names.index('construct_prompt')
        prompt = prompt_funcs[index](scraped_text, prompt_params['decomposed_justification'])
        with get_openai_callback() as callback_handler:
            response = llm.invoke(prompt)
    else:
        index = func_names.index('construct_mapreduce_prompt')
        prompt = prompt_funcs[index](prompt_params['decomposed_justification'])
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

    return response

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
                prompt_params={'decomposed_justification':decomposed_justification}
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
                    page_info['justification_summary'] = summarize_justification(llm, scrapped_text, start_time, prompt_params=prompt_params)         
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
