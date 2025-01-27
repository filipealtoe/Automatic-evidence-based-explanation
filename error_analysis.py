import argparse
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import ttest_1samp, f_oneway, kruskal, chi2_contingency

def clean_claims(dataframe, claim_column):
    claims = dataframe[claim_column]
    try:
        claims_ = claims._values
        clean_claims_ = []
        for claim_ in claims_:
            clean_claims = []
            for claim in claim_:
                claim = claim.replace('“', '').replace('”', '')
                clean_claims.append(claim)
            clean_claims_.append(clean_claims)
        dataframe[claim_column] = clean_claims_
    except:
        clean_claims = []
        for claim in claims:
            claim = claim.replace('“', '').replace('”', '')
            clean_claims.append(claim)
        dataframe[claim_column] = clean_claims
    return dataframe

def retrieve_artifacts(categories, stats_df_, classification_df, label = 'skip'):
    all_artifacts = []
    if label != 'skip':
        classification_df_ = classification_df.loc[classification_df['label'] == label]
        stats_df = stats_df_.loc[stats_df_['Class'] == label]
    else: 
        classification_df_ = classification_df
        stats_df = stats_df_
    
    claims = stats_df.loc[stats_df['category'].isin(categories)]
    #claims = clean_claims(claims, 'Claims')
    for i in range(0, claims.shape[0]):
        artifact = {}
        all_claims = claims.iloc[i]['claims']
        claims_to_include = []
        for single_claim in all_claims:
            claims_to_include.append(single_claim['claim'])
        decomposed_search_hits = classification_df_.loc[classification_df_['claim'].isin(claims_to_include)]['decomposed_search_hits'] 
        all_decomposition = []
        for hits in decomposed_search_hits:
            all_hits = []
            for hit in hits:
                decomposition = {}
                decomposition['decomposed_question'] = hit['decomposed_question']
                decomposition['decomposed_justification'] = hit['decomposed_justification']
                decomposition['decomposed_question_explanation'] = hit['decomposed_justification_explanation']
                all_evidence = []
                for page in hit['pages_info']:
                    evidence = {}
                    evidence['page_url'] = page['page_url']
                    evidence['page_content'] = page['page_content']
                    evidence['page_summary'] = page['justification_summary']['output_text']
                    evidence['page_timestamp'] = page['page_timestamp']
                    all_evidence.append(evidence)
                decomposition['evidence'] = all_evidence
                all_hits.append(decomposition)
            all_decomposition.append(all_hits)
        all_artifacts.append(all_decomposition)
    claims['artifacts'] = all_artifacts
    return claims

def general_error_data(args):
    #Chunking to handle very large datasets
    stats_df = read_large_json(args.stats_file_path)
    classified_df = read_large_json(args.summary_merged_path)
    #stats_df = pd.read_json(args.stats_file_path, lines=True)
    #classified_df = pd.read_json(args.summary_merged_path, lines=True)
    missclassified_categories = ['False Negatives', 'False Positives']
    correctlyclassified_categories = ['True Negatives', 'True Positives']
    retrieved_claims = retrieve_artifacts(missclassified_categories, stats_df, classified_df)
    retrieved_claims = pd.concat([retrieved_claims,retrieve_artifacts(correctlyclassified_categories, stats_df, classified_df)], axis=0)
    return retrieved_claims

def specific_error_data(label, categories, stats_df, classification_df):
    classified_claims = retrieve_artifacts(categories, stats_df, classification_df, label)
    return classified_claims

def get_array_params(string_param = ''):
    string_param = string_param.replace(', ', ',')
    if string_param == None or string_param == 'Skip' or string_param == 'skip':
        return('skip')
    try:
        string_param.remove('')
    except:
        pass
    parameters = string_param.split(',')
    return parameters

def read_large_json(dataset_path):
    def read_json_in_chunks(file_path, chunksize=1000):
        with open(file_path, 'r') as f:
            chunks = []
            for i, line in enumerate(f):
                chunks.append(pd.read_json(line, lines=True))
                if (i + 1) % chunksize == 0:
                    yield pd.concat(chunks)
                    chunks = []
            if chunks:
                yield pd.concat(chunks)
    chunks = []
    for chunk in read_json_in_chunks(dataset_path, chunksize=1000):
        chunks.append(chunk)
    merged = pd.concat(chunks, ignore_index=True)
    return merged

def main(args):
    if args.general_error:
        final_df = general_error_data(args)      
    
    if args.specific_error:
        categories = get_array_params(string_param = args.specific_categories)
        all_categories = []
        for category in categories:
            specific_category_claims = specific_error_data(args.specific_error_class, [category], stats_df, classified_df)
            all_categories.append(specific_category_claims)
        final_df = pd.concat(all_categories, axis=0)    

    #final_df.to_json(args.output_file_path, orient='records', lines=True)
    #Chunking to handle very large datasets
    with open(args.output_file_path, 'w') as f:
        for chunk in np.array_split(final_df, 100):  # Split into 100 smaller chunks
            chunk.to_json(f, orient='records', lines=True)
               
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_file_path', type=str, default=None)
    parser.add_argument('--summary_merged_path', type=str, default=None)
    parser.add_argument('--output_file_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--general_error', type=int, default=None)
    parser.add_argument('--specific_error', type=int, default=None)
    parser.add_argument('--specific_error_class', type=str, default=None)
    parser.add_argument('--specific_categories', type=str, default=None)
    args = parser.parse_args()
    main(args)