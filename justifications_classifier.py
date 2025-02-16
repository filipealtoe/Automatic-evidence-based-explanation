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
from LLMsummarizer import promptLLM


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

usage_log = deque()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("file_management.log"), logging.StreamHandler()]
)

# Format example for static prompt
def construct_mapreduce_prompt(decomposed_justification, prompt_params={}):
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


def construct_prompt(doc_text, prompt_params=None):
    old_prompt = ''' Given the following claim: "{}",

    context: "{}", 

    and hypothesis related to the claim: "{}",

    Use only the context provided to determine if the hypothesis is True, False or Unverified.  
    Your response should be a single word.
'''.format(prompt_params['claim'], doc_text, prompt_params['decomposed_justification'])
    
    prompt = ''' Given the following context: "{}".

    Use only the context provided to verify of the following hypothesis is True, False or Unverified.  
    Hypothesis: "{}'
    Your response should be a single word.
'''.format(doc_text, prompt_params['decomposed_question'])
    
    prompt = ''' Given the following context: "{}".

    Use only the context provided to answer the following question as Yes, No or Unverified.  
    Question: "{}'
    Your response should be a single word.
'''.format(doc_text, prompt_params['decomposed_question'])
    
    return prompt

func_prompts = [construct_prompt, construct_mapreduce_prompt]

def main(args):
    df = pd.read_json(args.input_path, lines=True)
    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end

    #Temperature = 0 as we want the summary to be factual and based on the input text
    llm = ChatOpenAI(temperature = 0, model = ENGINE, api_key = api_key, max_tokens = 1024, max_retries = MAX_GPT_CALLS)
    start_time = time.time()
    all_rows = []
    for i in tqdm(range(start, end)):
        try:
            decomposed_search_hits = df.iloc[i]['decomposed_search_hits']
            claim = df.iloc[i]['claim']
            label = df.iloc[i]['label']
            row_info = {'claim': claim, 'label':label}            
            j = 0
            for decomposed_search_hit in decomposed_search_hits:
                row_info['decomposed_justification'] = decomposed_search_hit['decomposed_justification']
                row_info['decomposed_question'] = decomposed_search_hit['decomposed_question']
                row_info['decomposed_justification_explanation'] = decomposed_search_hit['decomposed_justification_explanation']       
                prompt_params = {'decomposed_justification':row_info['decomposed_justification'], 'claim':claim,
                                 'decomposed_question':row_info['decomposed_question']}
                response = promptLLM(llm, func_prompts, row_info['decomposed_justification_explanation'], 
                                     start_time=start_time, prompt_params=prompt_params)                
                row_info['justification_explanation_verdict'] = response.content
                all_rows.append(row_info.copy())
                if j == 0:
                    row_info['claim'] = None
                    row_info['label'] = None
                decomposed_search_hit['justification_explanation_verdict'] = row_info['justification_explanation_verdict']
                j = j + 1
        except Exception as e:
            print("error caught", e)
            print('Dataset row = ', i)
            print('Pages Info index = ', j)
            print('Decomposed Question: ', decomposed_search_hit['decomposed_question'])
            print('Decomposed Justification: ', decomposed_search_hit['decomposed_justification'])          

    
    df.to_json(args.output_path, orient='records', lines=True)
    #Auxilary file to facilitate manual analysis
    df1 = pd.DataFrame(all_rows)
    csv_file_path = args.output_path.split('.jsonl')[0] + '.csv'
    df1.to_csv(csv_file_path, sep ='\t', header=True, index=False, encoding='utf-8')
    print('Done classifying!!!') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    main(args)
