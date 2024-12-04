import argparse
import os
import logging
from collections import deque
import time
import claim_decomposer, web_search, justification_summarizer_approach1, justification_summarizer_approach2
import justification_summaries_merger, justifications_classifier, evidence_based_article


# OpenAI API Key
#openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")

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
    handlers=[logging.FileHandler("file_management.log"), logging.StreamHandler()]
)


def main(args):
    try:
        args.output_path = args.decomposition_output_path
        print('Start Ddecomposition!!!')
        start_time = time.time()
        run_start_time = time.time()
        claim_decomposer.main(args)
        print('Time to complete Decomposition (sec): {}'.format(str(time.time() - start_time)))
        print('Decomposition Done!!!')
        args.input_path = args.decomposition_output_path
        args.output_path = args.web_search_output_path
        start_time = time.time()
        web_search.main(args)
        print('Time to complete Web Search (sec): {}'.format(str(time.time() - start_time)))
        print('All Web Search Done!!!')
        args.input_path = args.web_search_output_path
        args.output_path = args.justification_summarization_output_path
        start_time = time.time()
        justification_summarizer_approach2.main(args)
        print('Time to complete Summarization (sec): {}'.format(str(time.time() - start_time)))
        print('Justification Summarization Done!!!')
        args.input_path = args.justification_summarization_output_path
        args.output_path = args.justification_merger_output_path
        start_time = time.time()
        justification_summaries_merger.main(args)
        print('Time to complete Summarization Merger (sec): {}'.format(str(time.time() - start_time)))
        print('Justification Merging Done!!!')
        args.input_path = args.justification_merger_output_path
        args.output_path = args.classifier_output_path
        start_time = time.time()
        justifications_classifier.main(args)
        print('Time to complete Classification (sec): {}'.format(str(time.time() - start_time)))
        print('All Done!!!') 
        print('Total Time to complete the Run (sec): {}'.format(str(time.time() - run_start_time)))
    except Exception as e:
        print("error caught", e)             
    
    


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
    args = parser.parse_args()
    main(args)
