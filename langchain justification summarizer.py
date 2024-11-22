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
#from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager
from langchain.callbacks import OpenAICallbackHandler

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("file_management.log"), logging.StreamHandler()]
)

# Format example for static prompt
def construct_prompt(decomposed_justification, scraped_text):
    prompt = ''' Hypothesis: {}
The following text may or may not provide evidence to the hypothesis. Summarize the text including only the passages that are relevant to confirm or deny the hypothesis.
{}
Return only the summary without any additional text.
'''.format(decomposed_justification, scraped_text)
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

def summarize_justification(llm, decomposed_justification, scraped_text, max_prompt_tokens = 4000):
    #Temperature = 0 as we want the summary to be factual and based on the input text
    num_tokens = llm.get_num_tokens(scraped_text)
    #If scraped text is smaller than a single prompt chunk, just run the simple prompt
    if num_tokens <= max_prompt_tokens:
        prompt = construct_prompt(decomposed_justification, scraped_text)
        response = llm(prompt)
    else:
        #chunk_size is in words. 1 token approximatelly 0.75 words
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=int(max_prompt_tokens/0.75), chunk_overlap=500) 
        docs = text_splitter.create_documents([scraped_text])
        prompt = construct_mapreduce_prompt(decomposed_justification)
        summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=prompt['map_prompt'],
                                     combine_prompt=prompt['combine_prompt'],
                                     #verbose=True
                                    )
        response = summary_chain.invoke(docs)
    return response


def main(args):
    df = pd.read_json(args.input_path, lines=True)
    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end

    #This is to monitor the llm tokens per minute to avoid errors
    callback_handler = OpenAICallbackHandler()
    callback_manager = CallbackManager([callback_handler])

    llm = ChatOpenAI(temperature = 0, model = ENGINE, api_key = api_key, max_tokens = 1024, max_retries = MAX_GPT_CALLS, callback_manager=callback_manager)

    usage_log = deque()

    for i in tqdm(range(start, end)):
        try:
            decomposed_search_hits = df.iloc[i]['decomposed_search_hits']
            for decomposed_search_hit in decomposed_search_hits:
                decomposed_justification = decomposed_search_hit['decomposed_justification']
                i = 0
                start_time = time.time()
                #Summarize each url page content
                for page_info in decomposed_search_hit['pages_info']:
                    page_info['justification_summary'] = summarize_justification(llm, decomposed_justification, page_info['page_content']) 
                    # Access token usage information
                    tokens_used = callback_handler.total_tokens
                    # Log the token usage with a timestamp
                    usage_log.append((start_time, tokens_used))
                    # Remove outdated entries from the log (older than the interval)
                    while usage_log and (time.time() - usage_log[0][0]) > INTERVAL_SECONDS:
                        usage_log.popleft()
                    # Calculate token usage in the last INTERVAL_SECONDS
                    tokens_in_interval = sum(tokens for timestamp, tokens in usage_log)
                    print(f"Tokens used in summarization: {tokens_used}")
                    print(f"Tokens in interval: {tokens_in_interval}")
                    if tokens_in_interval  > TOKEN_THRESHOLD:
                        print(f"Token usage exceeded {TOKEN_THRESHOLD} tokens. Pausing for {DELAY_SECONDS} seconds...")
                        time.sleep(DELAY_SECONDS)    
                        callback_handler.total_tokens = 0   
                        start_time = time.time()    
                    decomposed_search_hit['pages_info'][i] = page_info.copy()
                    i = i + 1
        except Exception as e:
            print("error caught", e)
            print('i=', i)

    df.to_json(args.output_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    main(args)
