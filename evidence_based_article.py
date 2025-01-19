import argparse
import os
import glob
import openai
import logging
import time
from collections import deque
from logging.handlers import RotatingFileHandler
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from LLMsummarizer import promptLLM
import numpy as np
from LLMsummarizer import faiss_similarity_index,faiss_similarity_index_old, Faiss_similarity_search



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

def merge_chunked_files(files_dir, consolidated_file_name, files_pattern):
    #df=pd.DataFrame()
    all_dfs = []
    jsonl_file = 0
    for file in glob.glob(f"{files_dir}/{files_pattern}"):
        #if file.endswith(files_extension):
        try:
            aux = pd.read_json(os.path.join(files_dir, file), lines=True)
            jsonl_file = 1
        except:
            aux=pd.read_csv(os.path.join(files_dir, file), encoding='utf-8', sep='\t', header=0)
        all_dfs.append(aux)
    df = pd.concat(all_dfs, axis=0)
    if not jsonl_file:
        df.to_csv(os.path.join(files_dir, consolidated_file_name), sep ='\t', header=True, index=False, encoding='utf-8')
    else:
        df.to_json(os.path.join(files_dir, consolidated_file_name), orient='records', lines=True)

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

def drop_missing_human_articles(dataset, corpus_dataset):
    rows_to_drop = []
    for i in range(0,len(dataset)):
        human_article = corpus_dataset.loc[corpus_dataset['claim'] == dataset.iloc[i]['claim']]['human_article_text'].values[0]
        if human_article == '':
            rows_to_drop.append(i)
    dataset = dataset.drop(rows_to_drop)
    return dataset

func_prompts = [construct_prompt, construct_mapreduce_prompt]

def main(args):
    run_start_time = time.time()
    df = pd.read_json(args.input_path, lines=True)
    scraped_file_df = pd.read_json(args.scraped_file_path, lines=True)
    #Only generate articles for the rows that there is a human counterpart article for comparison
    df = drop_missing_human_articles(df, scraped_file_df)
    start = 0 if not args.start else args.start
    end = args.end
    if not args.end:
        end = len(df)
    else:
        if args.end > len(df):
            end = len(df)
    df = df[start:end]
    #Split dataset into chunk for intermediate file processing and saving
    claims_chunk = args.claims_chunk
    if df.shape[0]>=claims_chunk:
        if divmod(len(df),claims_chunk)[1] == 0:
            remainder = 0
        else:
            remainder = 1
        chunks = len(df)//claims_chunk + remainder
    else:
        chunks = 1
    data_path = args.final_output_path + '_' + time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    run_start_time = time.time()
    list_final_files_to_merge = []
    for chunk_ind in tqdm(range(0,chunks)):
        if chunk_ind != (chunks-1):
            df_final_ind = chunk_ind*claims_chunk + claims_chunk
        else:
            df_final_ind = df.shape[0]
        args.input_path = data_path + '/' + 'chunk' + str(chunk_ind) + '.jsonl'
        chunk_df = df.iloc[chunk_ind*claims_chunk:df_final_ind]
        chunk_df.to_json(args.input_path, orient='records', lines=True)
        args.output_path = data_path + '/' + 'evidence' + str(chunk_ind) + '.jsonl'
        df_out = pd.DataFrame()
        df_out['example_id'] = chunk_df['example_id']
        df_out['claim'] = chunk_df['claim']
        df_out['human_label'] = chunk_df['label']
        df_out['LLM_decomposing_prompt'] = chunk_df['LLM_decomposing_prompt']
        df_out['generated_label'] = np.nan
        #Temperature = 0 as we want the summary to be factual and based on the input text
        llm = ChatOpenAI(temperature = 0, model = ENGINE, api_key = api_key, max_tokens = 1024, max_retries = MAX_GPT_CALLS)        
        evidence_based_articles = []
        human_articles = []
        human_summaries = []
        article_similarities = []
        article_summaries = []
        article_summaries_similarities = []
    
        for i in tqdm(range(0, len(chunk_df))):
            explanations = []
            decomposed_search_hits = chunk_df.iloc[i]['decomposed_search_hits']           
            for decomposed_search_hit in decomposed_search_hits:
                explanation = {}
                explanation['decomposed_question'] = decomposed_search_hit['decomposed_question']
                explanation['decomposed_justification'] = decomposed_search_hit['decomposed_justification']            
                explanation['decomposed_explanation'] = decomposed_search_hit['decomposed_justification_explanation']
                justifications = []
                start_time = time.time()
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
            scraped_text = scraped_file_df.loc[scraped_file_df['claim'] == chunk_df.iloc[i]['claim']]['human_article_text'].values[0]
            human_articles.append(scraped_text)
            human_summary = scraped_file_df.loc[scraped_file_df['claim'] == chunk_df.iloc[i]['claim']]['human_summary'].values[0]
            human_summaries.append(human_summary)

            #Only generates evidence article if dataset includes a human generated article - for comparison purposes
            if scraped_text != '':
                for sub_explanation in explanations:
                    explanation_ = explanation_ + '/n' + sub_explanation['decomposed_explanation']
                    decomposed_questions_ = decomposed_questions_ + '/n' + sub_explanation['decomposed_question']       
                prompt_params = {'claim':chunk_df.iloc[i]['claim'], 'label':df_out.iloc[i]['human_label']}
                response = promptLLM(llm, func_prompts, explanation_, max_prompt_tokens=8000,
                                        start_time=time.time(), 
                                        prompt_params=prompt_params)   
                try:
                    evidence_article = response.content         
                except:
                    evidence_article = response['output_text']
                evidence_based_articles.append(evidence_article)
                #Articles similarity comparison of articles
                try:
                    article_similarity = faiss_similarity_index_old(scraped_text, evidence_article, max_prompt_tokens=4000, token_limit=8191)
                except:
                    article_similarity = None
                article_similarities.append(article_similarity)

                #Summarization only if human summary exists - for comparison purposes
                if human_summary != '':
                    try:
                        evidence_article_summary = summarize_article(evidence_article, decomposed_questions_, df_out.iloc[i]['claim'], args)
                    except:
                        evidence_article_summary = ''
                    #Articles summaries similarity comparison of articles
                    try:
                        article_summary_similarity = faiss_similarity_index(human_summary, evidence_article_summary, max_prompt_tokens=4000, token_limit=8191)
                    except:
                        article_summary_similarity = None
                else:
                    evidence_article_summary = ''
                    article_summary_similarity = None
                article_summaries.append(evidence_article_summary)
                article_summaries_similarities.append(article_summary_similarity)
            else:
                evidence_based_articles.append("")
                article_similarities.append(None)
                article_summaries.append('')
                article_summaries_similarities.append(None)
        
        df_out['human_article'] = human_articles 
        df_out['generated_article'] = evidence_based_articles   
        df_out['article_similarity'] = article_similarities 
        df_out['human_summary'] = human_summaries
        df_out['generated_article_summary'] = article_summaries   
        df_out['article_summary_similarity'] = article_summaries_similarities 
        
        df_out.to_json(args.output_path, orient='records', lines=True)
        print('Done generating Chunk{}'.format(chunk_ind))

    merge_chunked_files(data_path, 'merged_article_generation.jsonl', files_pattern='evidence*.jsonl')
    print('Done generating!!!') 
    print('Total Time to complete the Run (sec): {}'.format(str(time.time() - run_start_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--final_output_path', type=str, default=None)
    parser.add_argument('--scraped_file_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--claims_chunk', type=int, default=20)
    args = parser.parse_args()
    main(args)
