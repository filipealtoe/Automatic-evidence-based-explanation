from bs4 import BeautifulSoup
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

def scrape_article(url):
    article = []
    shortontime = []
    webpage = requests.get(url)  #Make a request to the website
    soup = BeautifulSoup(webpage.text, "html.parser") #Parse the text from the website
    try:
        article_text =  soup.find('article',attrs={'class':'m-textblock'})  #Get the tag and it's class
        shortontime_text =  soup.find('div',attrs={'class':'short-on-time'})
    except:
        return('','')
    try:
        if article_text!=None:
            for line in article_text:
                article.append(line.text)
            article_ = ' '.join(article)
        else:
            article_ = ''
    except:
        article_ = ''
    
    try:
        if shortontime_text!=None and args.get_summary:
            for line in shortontime_text:
                shortontime.append(line.text)
            shortontime_ = ' '.join(shortontime).replace("\n", "")
        else:
            shortontime_ = ''
    except:
        shortontime_ = ''    
    
    return article_, shortontime_


def main(args):
    df = pd.read_json(args.corpus_file_path, lines=True)
    start = 0 if not args.start else args.start
    if not args.end:
        end = len(df)
    else:
        end = args.end
        if args.end > len(df):
            end = len(df)
    output_df = df.copy()
    article_text =[]
    summaries = []
    for i in tqdm(range(end)):
        article, summary = scrape_article(df.iloc[i]['url'])
        '''except Exception as e:
            print(e)
            article = ''
            summary = ''
            '''
        article_text.append(article)
        summaries.append(summary)
    output_df['human_article_text'] = article_text
    output_df['human_summary'] = summaries
    output_df.to_json(args.dataset_output_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file_path', type=str, default=None)
    parser.add_argument('--dataset_output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--get_summary', type=int, default=None)
    args = parser.parse_args()
    main(args)