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
import numpy as np
from LLMsummarizer import faiss_similarity_index, Faiss_similarity_search



# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY_CIMPLE")

# OpenAI Engine, feel free to change
ENGINE = 'gpt-4o-mini'
# Max number of LLM retries
MAX_GPT_CALLS = 5

if ENGINE == 'gpt-4o-mini':
    max_tokens_min = 10000000
else:
    max_tokens_min = 2000000

TOKEN_THRESHOLD = int(0.75 * max_tokens_min)
INTERVAL_SECONDS = 45  # Time interval to monitor
DELAY_SECONDS = 60 #Delay if token_threshold is reached
#This will be used for the sumarization of generated article
NUMB_WORDS_PER_DOC = 400

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

    map_prompt = """You will be given text that will be enclosed in triple triple backquotes (''').
    Rewrite the following text in the format of an article without a title. 
    '''{text}''' 
    Your answer should return only the article and no other text."""
    
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt_half1 = """
    You will be given a series of summaries text. The text will be enclosed in triple backquotes (''')
    You are a fact-check article writer. Rewrite the following text in the format of an article without a title.
    '''{text}'''"""
    combine_prompt_half2 = '''
    Your answer should return only the article including a conclusion why the following claim is {}: {}'''.format(prompt_params['label'], prompt_params['claim'])
    combine_prompt = combine_prompt_half1 + combine_prompt_half2
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    prompt['map_prompt'] = map_prompt_template
    prompt['combine_prompt'] = combine_prompt_template
    return prompt


def construct_prompt(doc_text, prompt_params=None):
    prompt = ''' You are a fact-check article writer. Rewrite the following text in the format of an article without a title. "{}"
    Your answer should return only the article including a conclusion why the following claim is {}: {}
    
'''.format(doc_text, prompt_params['label'], prompt_params['claim'])
    return prompt


def summarize_article(article, decomposed_questions, claim, args):
    #The similiary is done by passing the decomposed questions to form the embeddings for similarity with article
    decomposed_questions_str = '/n'.join(decomposed_questions)
    numb_tokens = int(NUMB_WORDS_PER_DOC/0.75)  
    numb_docs = 1  
    #This is the text that will be used as query for similariy search against article text
    prompt_params = {'decomposed_justification':claim}
    try:           
        summary = Faiss_similarity_search(article, decomposed_questions_str, args, max_prompt_tokens = numb_tokens, 
                                                prompt_params=prompt_params, numb_similar_docs=1)
    except:
        summary = None
    return summary

func_prompts = [construct_prompt, construct_mapreduce_prompt]

def main(args):
    run_start_time = time.time()
    df = pd.read_json(args.input_path, lines=True)
    scraped_file_df = pd.read_json(args.scraped_file_path, lines=True)
    start = 0 if not args.start else args.start
    if not args.end:
        end = len(df)
    else:
        if args.end > len(df):
            end = len(df)
    df = df[start:end]
    df_out = pd.DataFrame()
    df_out['example_id'] = df['example_id']
    df_out['claim'] = df['claim']
    df_out['human_label'] = df['label']
    df_out['LLM_decomposing_prompt'] = df['LLM_decomposing_prompt']
    df_out['generated_label'] = np.nan
    #Temperature = 0 as we want the summary to be factual and based on the input text
    llm = ChatOpenAI(temperature = 0, model = ENGINE, api_key = api_key, max_tokens = 1024, max_retries = MAX_GPT_CALLS)
    explanations = []
    evidence_based_articles = []
    human_articles = []
    human_summaries = []
    article_similarities = []
    article_summaries = []
    article_summaries_similarities = []
    
    for i in tqdm(range(0, len(df))):
        decomposed_search_hits = df.iloc[i]['decomposed_search_hits']
        row_info = {}            
        j = 0
        for decomposed_search_hit in decomposed_search_hits:
            explanation = {}
            explanation['decomposed_question'] = decomposed_search_hit['decomposed_question']
            explanation['decomposed_justification'] = decomposed_search_hit['decomposed_justification']            
            explanation['decomposed_explanation'] = decomposed_search_hit['decomposed_justification_explanation']
            justifications = []
            start_time = time.time()
            k = 0
            evidences = []
            for page_info in decomposed_search_hit['pages_info']:
                if page_info['page_url'] == 'placeholder' or page_info['justification_summary']['output_text'] == 'placeholder':
                    continue
                evidences.append({'evidence_url':page_info['page_url'], 'evidence_summary':page_info['justification_summary']['output_text']})  
            explanation['evidence'] = evidences
            explanations.append(explanation)

        #Evidence based article generation
        explanation_ = ''
        decomposed_questions_ = ''
        for sub_explanation in explanations:
            explanation_ = explanation_ + '/n' + sub_explanation['decomposed_explanation']
            decomposed_questions_ = decomposed_questions_ + '/n' + sub_explanation['decomposed_question']       
        prompt_params = {'claim':df.iloc[i]['claim'], 'label':df_out.iloc[i]['human_label']}
        response = promptLLM(llm, func_prompts, explanation_, max_prompt_tokens=8000,
                                start_time=time.time(), 
                                prompt_params=prompt_params)   
        try:
            evidence_article = response.content         
        except:
            evidence_article = response['output_text']
        evidence_based_articles.append(evidence_article)
        scraped_text = scraped_file_df.loc[scraped_file_df['claim'] == df.iloc[i]['claim']]['human_article_text'].values[0]
        human_articles.append(scraped_text)
        #Articles similarity comparison of articles
        try:
            article_similarity = faiss_similarity_index(scraped_text, evidence_article, max_prompt_tokens=4000, token_limit=8191)
        except:
            article_similarity = -1
        article_similarities.append(article_similarity)

        #Summarization
        human_summary = scraped_file_df.loc[scraped_file_df['claim'] == df.iloc[i]['claim']]['human_summary'].values[0]
        human_summaries.append(human_summary)
        try:
            evidence_article_summary = summarize_article(evidence_article, decomposed_questions_, df_out.iloc[i]['claim'], args)
        except:
            evidence_article_summary = None
        article_summaries.append(evidence_article_summary)
        #Articles summaries similarity comparison of articles
        if human_summary != None:
            try:
                article_summary_similarity = faiss_similarity_index(human_summary, evidence_article_summary, max_prompt_tokens=4000, token_limit=8191)
            except:
                article_summary_similarity = None
        else:
            article_summary_similarity = None
        article_summaries_similarities.append(article_summary_similarity)
    
    df_out['human_article'] = human_articles 
    df_out['generated_artitle'] = evidence_based_articles   
    df_out['artitle_simiarity'] = article_similarities 
    df_out['human_summary'] = human_summaries
    df_out['generated_article_summary'] = article_summaries   
    df_out['artitle_summary_simiarity'] = article_summaries_similarities 
    
    df_out.to_json(args.output_path, orient='records', lines=True)
    print('Done generating!!!') 
    print('Total Time to complete the Run (sec): {}'.format(str(time.time() - run_start_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--scraped_file_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    main(args)
