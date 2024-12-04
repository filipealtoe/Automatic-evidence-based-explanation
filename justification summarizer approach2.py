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

#Semantic Similarity Returned Number of Words
NUMB_SIMILAR_WORDS_RETURNED = 1200
#Number of words per document
NUMB_WORDS_PER_DOC = 300


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

def main(args):
    '''In this approach the returned summaries is a merger of the top X docs returned from the semantic similarity against the justification. 
    This approach took 2h 23 mins and about $2,63 (GPT-4o-mini) for the 10 justifications and 10 URL docs per justification
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
                # Summarizing based on the justification
                #decomposed_justification = decomposed_search_hit['decomposed_justification']
                #prompt_params = {'decomposed_justification':decomposed_justification}
                # Summarizing based on the claim 
                decomposed_question = decomposed_search_hit['decomposed_question']
                prompt_params = {'decomposed_justification':decomposed_question}
                j = 0
                start_time = time.time()
                #Summarize each url page content
                for page_info in decomposed_search_hit['pages_info']: 
                    page_info['justification_summary'] = {}   
                    numb_tokens = int(NUMB_WORDS_PER_DOC/0.75)  
                    numb_docs = int(NUMB_SIMILAR_WORDS_RETURNED/NUMB_WORDS_PER_DOC)             
                    page_info['justification_summary']['output_text'] = Faiss_similarity_search(page_info['page_content'], args, max_prompt_tokens = numb_tokens, 
                                                               prompt_params=prompt_params, numb_similar_docs=numb_docs)
                    #If semantic similarity search took longer than the set interval for tokens per minute, restart start time
                    if time.time() - start_time > INTERVAL_SECONDS:
                        start_time = time.time()         
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
    print('Summarization Complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    main(args)
