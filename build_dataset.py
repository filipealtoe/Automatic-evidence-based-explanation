import requests
import os
import argparse
import pandas as pd
import uuid
import scrapy

api_key = os.getenv("GOOGLE_API_KEY")

MONTH_MAPPING = {"01": "January",
                 "1": "January",
                 "02": "February",
                 "2": "February",
                 "03": "March",
                 "3": "March",
                 "04": "April",
                 "4": "April",
                 "05": "May",
                 "5": "May",
                 "06": "June",
                 "6": "June",
                 "07": "July",
                 "7": "July",
                 "08": "August",
                 "8": "August",
                 "09": "September",
                 "9": "September",
                 "10": "October",
                 "11": "November",
                 "12": "December"
                 }

def query_fact_checked_claims(api_key, query, numb_claims=10, site_filter=None, 
                              language_code="en", max_age_days=None, claimant=None, offset=0):
    """
    Queries the Google Fact Check Tools API with various filters.
    
    Args:
        api_key (str): Your API key for the Google Fact Check Tools API.
        query (str): The query term to search for fact-checked claims.
        page_size (int): The maximum number of results to return (default is 10).
        site_filter (str): The specific fact-checker site to filter results from.
        language_code (str): The language code (default is 'en' for English).
        max_age_days (int): Max age of claims in days (optional).
        claimant (str): Filter by the claimant's name or organization (optional).
        offset (int): Number of results to skip for pagination (default is 0).
        
    Returns:
        dict: The response from the API.
    """
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "pageSize": numb_claims,
        "languageCode": language_code,
        "key": api_key,
        "offset": offset
    }
    
    if site_filter:
        params["reviewPublisherSiteFilter"] = site_filter
    if max_age_days is not None:
        params["maxAgeDays"] = max_age_days
    if claimant:
        params["claimant"] = claimant
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying the API: {e}")
        return None

def format_claim_date(claim_date = ''):
        month_day_year = claim_date.split('T')[0].split('-')
        date_str = "stated on {} {}, {}".format(MONTH_MAPPING[month_day_year[1]], month_day_year[2],  month_day_year[0])
        return date_str

def convert_labels(df, datasettype = 'snopes'):
    converted_df = pd.DataFrame()
    if datasettype == 'snopes':
        true_list = ['true', 'correct attribution']
        false_list = ['false']
        halftrue_list = ['half true', 'half-true', 'mixture']
        barelytrue_list = ['mostly false','mostly-false']
        mostlytrue_list = ['mostly true','mostly-true']
        for row in df.iterrows():
            df.iloc[row[0]]['label'] = df.iloc[row[0]]['label'].lower()
            append_row = 1
            if df.iloc[row[0]]['label'] in barelytrue_list:
                df.iloc[row[0]]['label'] = 'barely-true'
            elif df.iloc[row[0]]['label'] in false_list:
                df.iloc[row[0]]['label'] = 'false'
            elif df.iloc[row[0]]['label'] in halftrue_list:
                df.iloc[row[0]]['label'] = 'half-true'
            elif df.iloc[row[0]]['label'] in mostlytrue_list:
                df.iloc[row[0]]['label'] = 'mostly-true'
            elif df.iloc[row[0]]['label'] in true_list:
                df.iloc[row[0]]['label']= 'true'
            else:
                append_row = 0
            if append_row:
                converted_df = converted_df._append(df.iloc[row[0]])

    return converted_df

def main():
    '''
    from factcheckexplorer.factcheckexplorer import FactCheckLib

    # Initialize the library with your query and desired settings
    fact_check = FactCheckLib(query="Politics", language="en", num_results=1000)

    # Fetch the data
    fact_check.process()
    '''

    df = pd.DataFrame() 
    # Input parameters
    query = "Politicians"
    
    # Query the API
    '''
    results = []
    for offset in range(0, args.number_of_claims, 100):
        results_paginated = query_fact_checked_claims(api_key, query, args.number_of_claims, args.fact_checker, max_age_days=args.max_age_days)
        results.append(results_paginated)
    '''
    query = "Politics"
    results = query_fact_checked_claims(api_key, query, args.number_of_claims, args.fact_checker, max_age_days=args.max_age_days)
    example_ids = []
    labels = []
    urls = []
    claim_texts = []
    categories = []
    persons = []
    venues = []
    fact_checkers = []

    if results:
        claims = results.get("claims", [])
        if not claims:
            print("No fact-checked claims found.")
        else:
            for i, claim in enumerate(claims, start=1):
                print('Processing Claim Number {}', i)
                skip_claim = 0
                claim_text = claim.get("text", None)
                if claim == None:
                    continue
                person = claim.get("claimant", "")
                claim_review = claim.get("claimReview", [])               
                if claim_review:
                    for review in claim_review:
                        publisher = review.get("publisher", {}).get("name", None)
                        url = review.get("url", "No URL available")
                        #title = review.get("title", "No title available")
                        claim_date = review.get("reviewDate", None)
                        if claim_date == None:
                            continue
                        claim_label = review.get("textualRating", None)
                        if claim_label == None:
                            continue
                        #print(f"  Review by {publisher}: {title} - {url}")
                else:
                    continue
                example_ids.append(str(uuid.uuid4()))
                labels.append(claim_label)
                urls.append(url)
                claim_texts.append(claim_text)
                persons.append(person)
                venues.append(format_claim_date(claim_date))
                fact_checkers.append(publisher)
                categories.append(query)
            df['example_id'] = example_ids
            df['label'] = labels
            df['url'] = urls
            df['claim'] = claim_texts
            df['person'] = persons
            df['category'] = categories
            df['venue'] = venues
            df['fact_checker'] = fact_checkers
            df1 = convert_labels(df)
            df1.to_json(args.output_path, orient='records', lines=True)
    else:
        print("Failed to retrieve data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--fact_checker', type=str, default=None)
    parser.add_argument('--number_of_claims', type=int, default=None)
    parser.add_argument('--max_age_days', type=int, default=0)
    args = parser.parse_args()
    main()
