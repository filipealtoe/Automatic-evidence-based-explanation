import os
import time
import signal
import requests
from htmldate import find_date
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
import mmPeriodicTimer as timeout_alarm
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
                  "factcheck",
                  "wikipedia.org",
                  "facebook.com"
                  ]

PLACE_HOLDER = {
    "entities_info": [
        {
            "name": "placeholder",
            "description": "placeholder"
        }
    ],
    "pages_info": [
        {
            'page_name': 'placeholder',
            'page_url': 'placeholder',
            'page_timestamp': 'placeholder',
            'page_snippet': 'placeholder',
            'page_content':'placeholder'
        }
    ]
}

#This is called when class mmPeriodicTime ticks
def timeout_handler():
    print("received time out alarm")
    raise Exception("Web scraping time out!")

def scrape_website_text(url):
        logging.info(f"Starting to scrape URL: {url}")
        scraper_alarm = timeout_alarm.mmPeriodicTimer(interval=60, tickfunc=timeout_handler, periodic=False)
        scraper_alarm.start()
        try:
            # Fetch the content of the webpage
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Raise HTTPError for bad responses
            logging.info("Page fetched successfully.")

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            logging.debug("HTML content parsed with BeautifulSoup.")

            # Extract and concatenate visible text from the HTML elements
            text = ' '.join(element.get_text(strip=True) for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span', 'div']))
            logging.info(f"Successfully scraped text from URL: {url}")
            scraper_alarm.stop()
            return text

        except requests.RequestException as e:
            logging.error(f"An error occurred while fetching the URL: {e}")
            scraper_alarm.stop()
            return ""
        
        except requests.exceptions as e:
            logging.error(f"An error occurred while fetching the URL: {e}")
            scraper_alarm.stop()
            return ""

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            scraper_alarm.stop()
            return ""


class WebRetriever:

    def __init__(self, engine: str, sites_constrain=True, answer_count: int = 10):
        self.engine = engine
        if engine == 'bing':
            #self.subscription_key = os.environ[
                #'BING_SEARCH_V7_SUBSCRIPTION_KEY']
            self.subscription_key = "4eb55afbffd34ae9943c95a15da944b0"
            #self.endpoint = os.environ['BING_SEARCH_V7_ENDPOINT']
            self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
            self.mkt = 'en-US'
            self.answer_count = answer_count
            self.headers = {'Ocp-Apim-Subscription-Key': self.subscription_key}
            self.site_constraint = sites_constrain
        elif engine == 'google':
            pass

    logging.basicConfig(
    level=logging.INFO,                     # Set logging level to INFO or DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[logging.FileHandler("web_scraper.log"),   # Log to a file
              logging.StreamHandler()]                  # Print logs to the console
)
    
    def format_query(self, question, timestamp):
        #Set start search range to 5 years prior to claim date
        start_date_range = (datetime.strptime(timestamp, "%Y-%m-%d") - relativedelta(years=5)).strftime("%Y-%m-%d")
        exclusion_clause = " ".join(f"-site:{domain}" for domain in excluded_sites)
        query = f"{question} {exclusion_clause} after: {start_date_range} before:{timestamp}"
        return query


    def get_results(self, query: str, time_stamp=None, raw_count=30):
        if self.engine == 'bing':
            if time_stamp != None:
                start_date_range = (datetime.strptime(time_stamp, "%Y-%m-%d") - relativedelta(years=5)).strftime("%Y-%m-%d")
            params = {
                'q': query,
                'mkt': self.mkt,
                'count': raw_count,
                'freshness': "{}..{}".format(start_date_range, time_stamp) if
                time_stamp else None
            }
            try:
                start = time.time()
                response = requests.get(self.endpoint,
                                        headers=self.headers,
                                        params=params
                                        )
                response.raise_for_status()
                end = time.time()
                print(f'search for query:{query} finished, time take:', end - start)
            except Exception as e:
                print('exception happens during the search:')
                print(e)
                return PLACE_HOLDER
            response_json = response.json()
            entities_info = []
            pages_info = []

            if 'entities' in response_json.keys():
                entities_info = []
                for info in response_json['entities']['value']:
                    entities_info.append(
                        {
                            'name': info['name'],
                            'description': info['description']
                        }
                    )
            if 'webPages' not in response_json:
                return PLACE_HOLDER
            retrieved_pages = response_json['webPages']['value']
            count = 0
           
            for i, page in enumerate(retrieved_pages):
                page_info_entry = {
                    'page_name': None,
                    'page_url': None,
                    'page_timestamp': None,
                    'page_snippet': None,
                    'page_content': None
                }
                if not self.site_constraint or not self.url_is_excluded(page['url']):
                    page_info_entry['page_name'] = page['name']
                    page_info_entry['page_url'] = page['url']
                    page_info_entry['page_snippet'] = page['snippet']
                    

                    try:
                        # print("getting page timestamp:", page['url'])
                        page_date = find_date(page['url'], verbose=False, original_date=False)
                        page_info_entry['page_timestamp'] = page_date
                        # print(page_date)
                    except Exception as ex:
                        page_date = None
                        print('Error happens during extracting page timestamp:')
                        print(ex)
                    #If was able to retrieve page date, check it againts the timestamp and skip it if later date
                    if page_date != None: 
                        if (datetime.strptime(time_stamp, "%Y-%m-%d") < datetime.strptime(page_date, "%Y-%m-%d")):
                            continue
                    page_info_entry['page_content'] = scrape_website_text(page['url'])
                    if page_info_entry['page_content'] == "":
                        continue
                    pages_info.append(page_info_entry)
                    count += 1
                    if count == self.answer_count:
                        break
            return {
                "pages_info": pages_info
            }

    @staticmethod
    def url_is_excluded(url):
        for ex_url in excluded_sites:
            if ex_url in url:
                return True
        return False

