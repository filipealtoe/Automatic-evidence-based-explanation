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

usage_log = deque()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("file_management.log"), logging.StreamHandler()]
)

# Format example for static prompt
def construct_mapreduce_prompt(prompt_params={}):
    prompt = {}
    hypotesis = "Question: " + prompt_params['question']
    map_prompt = hypotesis + """

    You will be given text that may or may not answer the question above. The text will be enclosed in triple triple backquotes (''').
    Summarize the text including only the passages that are relevant to the question.
    '''{text}'''
    Return only the summary without any additional text.
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    You will be given a series of summaries text. The text will be enclosed in triple backquotes (''')
    '''{text}'''
    Summarize the text without losing information. Only include information that is present in the document in a factual manner.
    Your response should not make any reference to "the text" or "the document" and be ready to be merged into a fact-check article.
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    prompt['map_prompt'] = map_prompt_template
    prompt['combine_prompt'] = combine_prompt_template
    return prompt


def construct_prompt(doc_text, prompt_params={}):
    #Approach 1
    prompt_old = ''' Document: "{}"

Summarize the document in a single paragraph. Only include information that is present in the document in a factual manner.
Your response should not make any reference to "the text" or "the document" and be ready to be merged into a fact-check article.
'''.format(doc_text)
    
    #Approach 2
    prompt = ''' Document: "{}"
Summarize the document in a single paragraph answering the following question: "{}

Only include information that is present in the document in a factual manner. 

Your response should not make any reference to "the text", "the document" or "the question" and be ready to be merged into a fact-check article.
'''.format(doc_text, prompt_params['question'])
    

    return prompt

func_prompts = [construct_prompt, construct_mapreduce_prompt]

def main(args):
    df = pd.read_json(args.input_path, lines=True)
    start = 0 if not args.start else args.start
    if not args.end:
        end = len(df)
    else:
        if args.end > len(df):
            end = len(df)

    #Temperature = 0 as we want the summary to be factual and based on the input text
    llm = ChatOpenAI(temperature = 0, model = ENGINE, api_key = api_key, max_tokens = 1024, max_retries = MAX_GPT_CALLS)
    all_rows = []
    for i in tqdm(range(start, end)):
        decomposed_search_hits = df.iloc[i]['decomposed_search_hits']
        row_info = {}            
        j = 0
        justification_summary_line = 0
        for decomposed_search_hit in decomposed_search_hits:
            row_info['decomposed_justification'] = decomposed_search_hit['decomposed_justification']
            row_info['decomposed_question'] = decomposed_search_hit['decomposed_question']
            row_info['justification_summary'] = None
            row_info['summary_number_of_tokens'] = None
            justifications = []
            start_time = time.time()
            justification_summary_line = len(all_rows)
            k = 0
            for page_info in decomposed_search_hit['pages_info']:
                row_info['page_url'] = page_info['page_url']  
                if 'output_text' in page_info['justification_summary']:                 
                    row_info['page_justification_summary'] = page_info['justification_summary']['output_text']
                else:
                    row_info['page_justification_summary'] = ''
                justifications.append(row_info['page_justification_summary'])
                row_info['number_of_tokens'] = llm.get_num_tokens(row_info['page_justification_summary'])
                all_rows.append(row_info.copy())
                row_info['decomposed_justification'] = None
                row_info['decomposed_question'] = None 
                k = k + 1       
            if len(justifications)!= 0:
                merged_justification = ''.join(justifications)
                prompt_params = {'decomposed_justification':decomposed_search_hit['decomposed_justification'],
                                    'question': decomposed_search_hit['decomposed_question']}
                try:
                    if args.FAISS:
                        response = Faiss_similarity_search(scrapped_text=merged_justification, statement_to_compare=decomposed_search_hit['decomposed_question'], args=args, max_prompt_tokens = 1000/0.75, 
                                                                prompt_params=prompt_params, numb_similar_docs=1)
                        #response_LLM = promptLLM(llm, func_prompts, merged_justification, start_time=start_time, prompt_params=prompt_params)
                    else:
                        response = promptLLM(llm, func_prompts, merged_justification, start_time=start_time, prompt_params=prompt_params)
                    skip_response = 0
                except Exception as e:
                    response_text = ""
                    print("error caught", e)
                    print('Dataset row = ', i)
                    print('Pages Info index = ', j)
                    print('Page index = ', k)
                    print('Decomposed Question: ', decomposed_search_hit['decomposed_question'])
                    print('Decomposed Justification: ', decomposed_search_hit['decomposed_justification'])          
                    print('Page name: ', page_info['page_name']) 
                    print('Page url: ', page_info['page_url'])
                    skip_response = 1
                if not skip_response:
                    if type(response) == str:
                        response_text = response
                    else:
                        try:
                            response_text = response.content
                        except:
                            response_text = response['output_text']                    
                    skip_response = 0  
                #all_rows[justification_summary_line]['justification_summary'] = response_text
                #all_rows[justification_summary_line]['summary_number_of_tokens'] = llm.get_num_tokens(row_info['page_justification_summary'])   
                all_rows[-1]['justification_summary'] = response_text
                all_rows[-1]['summary_number_of_tokens'] = llm.get_num_tokens(response_text)        
                justifications = []
            else:
                response_text = 'No justification created'
            decomposed_search_hit['decomposed_justification_explanation'] = response_text
            j = j + 1
        
    
    df.to_json(args.output_path, orient='records', lines=True)
    #Auxilary file to facilitate manual analysis
    df1 = pd.DataFrame(all_rows)
    csv_file_path = args.output_path.split('.jsonl')[0] + '.csv'
    df1.to_csv(csv_file_path, sep ='\t', header=True, index=False, encoding='utf-8')
    print('Done merging!!!') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--FAISS', type=int, default=None)
    args = parser.parse_args()
    main(args)
