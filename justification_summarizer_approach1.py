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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from LLMsummarizer import promptLLM, semantic_similarity_search

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY_CIMPLE")

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

# Format example for static prompt
def construct_prompt(doc_text, prompt_params={}):
    prompt = ''' Hypothesis: {}
The following text may or may not provide evidence to the hypothesis. Summarize the text including only the passages that are relevant to confirm or deny the hypothesis.
{}
Return only the summary without any additional text.
'''.format(prompt_params['decomposed_justification'], doc_text)
    return prompt

# Format example for static prompt
def construct_mapreduce_prompt(decomposed_justification):
    prompt = {}
    hypotesis = "Hypothesis: " + decomposed_justification
    map_prompt = hypotesis + """

    You will be given text that may or may not provide evidence to the hypothesis. The text will be enclosed in triple triple backquotes (''').
    Summarize the text including only the passages that are relevant to confirm or deny the hypothesis.
    '''{text}'''
    Return only the summary without any additional text.
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    You will be given a series of summaries text. The text will be enclosed in triple backquotes (''')
    Summarize the text without losing information.
    '''{text}'''
    Return only the summary without any additional text.
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    prompt['map_prompt'] = map_prompt_template
    prompt['combine_prompt'] = combine_prompt_template
    return prompt

func_prompts = [construct_prompt, construct_mapreduce_prompt]


def main(args):
    '''In this approach we summarize each URL doc using a LLM. If the URL doc is larger than 0.75 * LLM max number of tokens per minute, 
    perform semantic similarity against the justification, then extract an arbitrary number (20) of similar docs as the summary. Otherwise,
    perform LLM summarization. This approach took 2h 23 mins and about $2,63 (GPT-4o-mini) for the 10 justifications and 10 URL docs per justification
    '''
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
                prompt_params = {'decomposed_justification':decomposed_justification}
                j = 0
                start_time = time.time()
                #Summarize each url page content
                for page_info in decomposed_search_hit['pages_info']:
                    num_tokens = llm.get_num_tokens(page_info['page_content'])
                    #If content is too large, do semantic similarity with justification to return only the relevant chunks
                    if num_tokens >= TOKEN_THRESHOLD:                        
                        scrapped_text = semantic_similarity_search(page_info['page_content'], args, max_prompt_tokens = 4000, prompt_params=prompt_params)
                        #If semantic similarity search took longer than the set interval for tokens per minute, restart start time
                        if time.time() - start_time > INTERVAL_SECONDS:
                            start_time = time.time()
                    else: 
                        scrapped_text = page_info['page_content']
                    page_info['justification_summary'] = promptLLM(llm, func_prompts, scrapped_text, start_time, prompt_params=prompt_params)         
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
