import argparse
import os
import logging
from collections import deque
import pathlib
import time
import claim_decomposer, web_search, justification_summarizer_approach1, justification_summarizer_approach2
import justification_summaries_merger, justifications_classifier, evidence_based_article, tavily_search
import pandas as pd
from tqdm import tqdm


# OpenAI API Key
#openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("OPENAI_API_KEY_CIMPLE")

# OpenAI Engine, feel free to change
ENGINE = 'gpt-4o'
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
    handlers=[logging.FileHandler("workflow.log"), logging.StreamHandler()]
)

def merge_chunked_files(files_dir, consolidated_file_name, files_extension='.csv'):
    #df=pd.DataFrame()
    all_dfs = []
    for file in os.listdir(files_dir):
        if file.endswith(files_extension):
            aux=pd.read_csv(os.path.join(files_dir, file), encoding='utf-8', sep='\t', header=0)
            all_dfs.append(aux)
    df = pd.concat(all_dfs, axis=0)
    df.to_csv(os.path.join(files_dir, consolidated_file_name), sep ='\t', header=True, index=False, encoding='utf-8')
    


def main(args):
    df = pd.read_json(args.input_path, lines=True)
    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end
    #Split dataset into four chunks of 20 claims for intermediate file saving
    claims_chunk = 20
    if df.shape[0]>=claims_chunk:
        chunks = [df[i:i + claims_chunk] for i in range(start, end, claims_chunk)]
    else:
        chunks = [df[start:end]]
    args.start = 0
    args.end = None
    data_path = args.final_output_path.split('.jsonl')[0] + '_' + time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        ####DELETE
    #merge_chunked_files(data_path,"all_classified.csv")
    i = 0
    run_start_time = time.time()
    list_final_files_to_merge = []
    for chunk in tqdm(chunks):
        try:
            args.input_path = data_path + '/' + 'chunk' + str(i) + '.jsonl'
            chunk.to_json(args.input_path, orient='records', lines=True)
            args.output_path = data_path + '/' + 'decomposed' + str(i) + '.jsonl'
            print('Start Decomposition!!!')
            start_time = time.time()
            claim_decomposer.main(args)
            print('Time to complete Decomposition (sec): {}'.format(str(time.time() - start_time)))
            print('Decomposition Done!!!')
            args.input_path = args.output_path
            args.output_path = data_path + '/' + 'websearch' + str(i) + '.jsonl'
            start_time = time.time()
            if not args.use_Tavily_search:                
                web_search.main(args)
                print('Time to complete Web Search (sec): {}'.format(str(time.time() - start_time)))
                print('All Web Search Done!!!')
                args.input_path = args.output_path
                args.output_path = data_path + '/' + 'summary' + str(i) + '.jsonl'
                start_time = time.time()
                justification_summarizer_approach2.main(args)
                print('Time to complete Summarization (sec): {}'.format(str(time.time() - start_time)))
                print('Justification Summarization Done!!!')
                args.input_path = args.output_path
                args.output_path = data_path + '/' + 'summarymerged' + str(i) + '.jsonl'
                start_time = time.time()
                justification_summaries_merger.main(args)
                print('Time to complete Summarization Merger (sec): {}'.format(str(time.time() - start_time)))
                print('Justification Merging Done!!!')
            else:
                tavily_search.main(args)
                print('Time to complete Web Search (sec): {}'.format(str(time.time() - start_time)))
                print('All Web Search Done!!!')
            args.input_path = args.output_path
            args.output_path = data_path + '/' + 'classification' + str(i) + '.jsonl'
            start_time = time.time()
            justifications_classifier.main(args)
            print('Time to complete Classification (sec): {}'.format(str(time.time() - start_time)))
            i = i + 1
            list_final_files_to_merge.append(args.output_path)
        except Exception as e:
            print("error caught", e)  
            logging.info(f"Error: {e}")   
            logging.info(f"Chunk: {i}") 
        df_list = []
    for final_file in list_final_files_to_merge:
        df_list.append(pd.read_json(final_file, lines=True))
    final_df = pd.concat(df_list, axis=0)
    final_df.to_json(data_path + '/' + 'complete_workflow' + '.jsonl', orient='records', lines=True)
    merge_chunked_files(data_path,"all_classified.csv")
    print('All Done!!!') 
    print('Total Time to complete the Run (sec): {}'.format(str(time.time() - run_start_time)))
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--final_output_path', type=str, default=None)
    parser.add_argument('--decomposition_output_path', type=str, default=None)
    parser.add_argument('--web_search_output_path', type=str, default=None)
    parser.add_argument('--justification_summarization_output_path', type=str, default=None)
    parser.add_argument('--justification_merger_output_path', type=str, default=None)
    parser.add_argument('--classifier_output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--use_time_stamp', type=int, default=1)
    parser.add_argument('--answer_count', type=int, default=10)
    parser.add_argument('--time_offset', type=int, default=1, help="add an offest to the time at which the claim was made on web search")
    parser.add_argument('--sites_constrain', type=int, default=1,
                        help="whether to constrain the web search to certain sites")
    parser.add_argument('--use_annotation', type=int, default=0, help="whether to use annotated questions on web search")
    parser.add_argument('--use_claim', type=int, default=0, help="whether to use claim as question on web search")
    parser.add_argument('--question_num', type=int, default=10, help="number of questions to use on web search")
    parser.add_argument('--use_Tavily_search', type=int, default=0, help="whether to use the Tavily intelligent search service or manual search")
    parser.add_argument('--chunk_size', type=int, default=4, help="size of the chunk to parallel process on web search")
    args = parser.parse_args()
    main(args)
