import argparse
import os
import re
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import logging

WH_MATCHES = ("why", "who", "which", "what", "where", "when", "how")

NUMBERS = ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,                     # Set logging level to INFO or DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[logging.FileHandler("web_scraper.log"),   # Log to a file
              logging.StreamHandler()]                  # Print logs to the console
)

def scrape_website_text(url: str) -> str:
    """
    Scrapes visible text from a given website URL.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        str: The scraped visible text or an empty string in case of an error.
    """
    logging.info(f"Starting to scrape URL: {url}")
    try:
        # Fetch the content of the webpage
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        logging.info("Page fetched successfully.")

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        logging.debug("HTML content parsed with BeautifulSoup.")

        # Extract and concatenate visible text from the HTML elements
        text = ' '.join(element.get_text(strip=True) for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span', 'div']))
        logging.info(f"Successfully scraped text from URL: {url}")

        return text

    except requests.RequestException as e:
        logging.error(f"An error occurred while fetching the URL: {e}")
        return ""

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return ""

def main(args):
    df = pd.read_json(args.input_path, lines=True)

    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end

    for i in tqdm(range(start, end)):
        search_hits = df.iloc[i]['decomposed_search_hits']
        search_hit_index = 0
        for search_hit in search_hits:
            pages_info = search_hit['pages_info']
            page_info_index = 0
            for page_info in pages_info:
                scraped_text = scrape_website_text(page_info['page_url'])
                page_info['scraped_text'] = scraped_text
                pages_info[page_info_index] = page_info
                page_info_index = page_info_index + 1
            search_hits[search_hit_index] = pages_info
            search_hit_index = search_hit_index + 1
        df.at[i, 'decomposed_search_hits'] = search_hits


    df.to_json(args.output_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--scrape_top_hits', type=int, default=10)
    args = parser.parse_args()
    main(args)
