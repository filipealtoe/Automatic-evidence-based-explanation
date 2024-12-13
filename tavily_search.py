import os
import time
import argparse
import pandas as pd
import models.raw_evidence_retriever as retriever
from tqdm import tqdm
from typing import Dict, Set
from datetime import datetime, timedelta
from multiprocessing import Pool
from tavily import TavilyClient

tavily_key = os.getenv("Tavily-Student-Key")
tavily_client = TavilyClient(api_key=tavily_key)

excluded_sites = ["politifact.com",
                  "factcheck.org",
                  "snopes.com",
                  "washingtonpost.com/news/fact-checker/",
                  "apnews.com/hub/ap-fact-check",
                  "fullfact.org/",
                  "reuters.com/fact-check",
                  "youtube.com",
                  ".pdf",
                  "fact-check",
                  "factcheck"
                  ]




def main(args):
    df = pd.read_json(args.input_path, lines=True)
    if args.start == None:
        start = 0
    else:
        start = args.start
    if args.end == None:
        end = len(df)
    else:
        end = min(len(df), args.end)

    df['decomposed_search_hits'] = ""
    start_time = time.time()
    try:
        for i, row in tqdm(df.iterrows()):
            questions = row['claim questions']
            questions = [q for q in questions if q.strip()]
            all_results = []
            #results = {
                #'entities_info': [],
                #'pages_info': []
            #}
            results = {}
            justification_index = 0
            for q in questions[:args.question_num]:
                results['decomposed_question'] = q
                results['decomposed_justification'] = row['justifications'][justification_index]
                justification_index = justification_index + 1
                res = tavily_client.search(query=q, search_depth="advanced", max_results=10, exclude_domains=excluded_sites, include_answer=True)
                results['pages_info'] = res['results']
                results['answer'] = res['answer']
                #update_results(results, res, all_urls, all_entity_names)
            #if not args.use_time_stamp:
                #chunk.at[i, 'search_results'] = results
            #else:
                #chunk.at[i, 'search_results_timestamp'] = results
                all_results.append(results.copy())
                print('Processing claim question: ', q)
            df.at[i, 'decomposed_search_hits'] = all_results

    except Exception as e:
        print('error:', e)
        print(f'current index = {i}, question = {q}')
        
    
    #Re-enable for sequential processing
    #chunks = df
    #results = process_chunk(chunks, len(chunks.index), web_retriever, args)
    #results.to_json(args.output_path, orient='records', lines=True)

    #Re-enable for parallel processing
    #chunks = [df[i:i + args.chunk_size] for i in range(start, end, args.chunk_size)]
    #with Pool() as pool:
        #process_arg = [(chunk, idx, web_retriever, args) for idx, chunk in enumerate(chunks)]
        #results = pool.starmap(process_chunk, process_arg)
    # Separate the results into two lists
    # Merge the results back into a single DataFrame
    #df_processed = pd.concat([r for r in results if r is not None])
    #df_processed.to_json(args.output_path, orient='records', lines=True)
    df.to_json(args.output_path, orient='records', lines=True)
    print('Web Search Total Time: ', time.time() - start_time)
    print('Done Web Searching!')
    return df
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None,
                        help="url of the input file, could be a local jsonl or a google sheet")
    parser.add_argument('--output_path', type=str, default=None, help="path of the output file")
    parser.add_argument('--use_time_stamp', type=int, default=1, help="whether to use time stamp as search constraint")
    parser.add_argument('--sites_constrain', type=int, default=1,
                        help="whether to constrain the search to certain sites")
    parser.add_argument('--use_annotation', type=int, default=0, help="whether to use annotated questions")
    parser.add_argument('--use_claim', type=int, default=0, help="whether to use claim as question")
    parser.add_argument('--question_num', type=int, default=10, help="number of questions to use")
    parser.add_argument('--answer_count', type=int, default=10, help="number of answers to retrieve")
    parser.add_argument('--start', type=int, default=0, help="start index of the data to do retrieval")
    parser.add_argument('--end', type=int, default=1000, help="end index of the data to do retrieval")
    parser.add_argument('--chunk_size', type=int, default=50, help="size of the chunk")
    parser.add_argument('--time_offset', type=int, default=1, help="add an offest to the time at which the claim was made")
    main(parser.parse_args())
