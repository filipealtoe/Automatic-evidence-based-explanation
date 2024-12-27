import argparse
import os
import re
import openai
import logging
import time
from collections import deque
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from LLMsummarizer import Faiss_similarity_search

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY_CIMPLE")

# OpenAI Engine, feel free to change
ENGINE = 'gpt-4o-mini'
# Max number of LLM retries
MAX_GPT_CALLS = 5

#Usage tier 4
if ENGINE == 'gpt-4o-mini':
    max_tokens_min = 10000000
else:
    max_tokens_min = 2000000

TOKEN_THRESHOLD = int(0.75 * max_tokens_min)
INTERVAL_SECONDS = 45  # Time interval to monitor
DELAY_SECONDS = 60 #Delay if token_threshold is reached

#Semantic Similarity Returned Number of Words
NUMB_SIMILAR_WORDS_RETURNED = 1200
#Number of words per document
NUMB_WORDS_PER_DOC = 300


usage_log = deque()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("justification summarizer.log"), logging.StreamHandler()]
)

def main(args):
    '''In this approach the returned summaries is a merger of the top X docs returned from the semantic similarity against the justification. 
    This approach took 2h 23 mins and about $2,63 (GPT-4o-mini) for the 10 justifications and 10 URL docs per justification
    '''
    df = pd.read_json(args.input_path, lines=True)
    start = 0 if not args.start else args.start
    if not args.end:
        end = len(df)
    else:
        if args.end > len(df):
            end = len(df)

    #Temperature = 0 as we want the summary to be factual and based on the input text
    llm = ChatOpenAI(temperature = 0, model = ENGINE, api_key = api_key, max_tokens = 1024, max_retries = MAX_GPT_CALLS)

    for i in tqdm(range(start, end)):
        decomposed_search_hits = df.iloc[i]['decomposed_search_hits']
        if decomposed_search_hits != '':
            for decomposed_search_hit in decomposed_search_hits:
                # Summarizing based on the justification
                #decomposed_justification = decomposed_search_hit['decomposed_justification']
                #prompt_params = {'decomposed_justification':decomposed_justification}
                # Summarizing based on the claim 
                decomposed_question = decomposed_search_hit['decomposed_question']
                prompt_params = {'decomposed_justification':decomposed_search_hit['decomposed_question'], 'decomposed_justification':decomposed_search_hit['decomposed_justification']}
                j = 0
                start_time = time.time()
                #Summarize each url page content
                for page_info in decomposed_search_hit['pages_info']: 
                    page_info['justification_summary'] = {}   
                    numb_tokens = int(NUMB_WORDS_PER_DOC/0.75)  
                    numb_docs = int(NUMB_SIMILAR_WORDS_RETURNED/NUMB_WORDS_PER_DOC)  
                    try:           
                        page_info['justification_summary']['output_text'] = Faiss_similarity_search(page_info['page_content'], decomposed_question, args, max_prompt_tokens = numb_tokens, 
                                                                prompt_params=prompt_params, numb_similar_docs=numb_docs)
                    except Exception as e:
                        page_info['justification_summary']['output_text'] = ""
                        print("error caught", e)
                        print('Dataset row = ', i)
                        print('Decomposed Question: ', decomposed_search_hit['decomposed_question'])
                        print('Decomposed Justification: ', decomposed_search_hit['decomposed_justification'])
                        print('Pages_Info index = ', j)           
                        print('Page name: ', page_info['page_name']) 
                        print('Page url: ', page_info['page_url'])
                        print('Page content length: ', len(page_info['page_content'].strip().split(" ")))
                    #If semantic similarity search took longer than the set interval for tokens per minute, restart start time
                    if time.time() - start_time > INTERVAL_SECONDS:
                        start_time = time.time()         
                    decomposed_search_hit['pages_info'][j] = page_info.copy()
                    j = j + 1

    df.to_json(args.output_path, orient='records', lines=True)
    print('Summarization Complete!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    main(args)
