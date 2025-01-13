import argparse
import os
import re
from datetime import datetime, timedelta
import pandas as pd
import time
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from LLMsummarizer import promptLLM

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY_CIMPLE")

WH_MATCHES = ("why", "who", "which", "what", "where", "when", "how")

NUMBERS = ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")

REGEX = "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?" \
            "|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?" \
            "|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})"

MONTH_MAPPING = {"Jan": "01",
                 "January": "01",
                 "Feb": "02",
                 "February": "02",
                 "Mar": "03",
                 "March": "03",
                 "Apr": "04",
                 "April": "04",
                 "May": "05",
                 "Jun": "06",
                 "June": "06",
                 "Jul": "07",
                 "July": "07",
                 "Aug": "08",
                 "August": "08",
                 "Sep": "09",
                 "September": "09",
                 "Oct": "10",
                 "October": "10",
                 "Nov": "11",
                 "November": "11",
                 "Dec": "12",
                 "December": "12"
                 }

# OpenAI Engine, feel free to change
ENGINE = 'gpt-4o'
# The stop criteria for question generation, feel free to change
MAX_GPT_CALLS = 5
#Number of decomposed questions to be generated
MAX_NUM_QUESTIONS = 10


def construct_prompt(claim, prompt_params=None):
    prompt = '''I will give you a series of claims and their corresponding categories:
                    Claim: "Elizabeth Warren lives in a multi-million-dollar mansion and relied on scant Native American heritage claims to land a job at Harvard."
                    Category: Politicians
                    Claim: "Did Bill and Hillary Clinton return furniture they took from the White House in 2001?"
                    Category: Politicians
                    Claim: "Immigration and Customs Enforcement  detained and deported Polish doctor Lukasz Niec, 40 years after he emigrated to the United States as a young child."
                    Category: Immigration
                    Claim: "One study that just came out looked at the prison population in Arizona and found that illegal aliens are more than twice as likely to be convicted of crimes as Arizonans."
                    Category: Immigration
                    Claim: "The Republican tax plan will be the largest tax cut in U.S. history."
                    Category: Politics
                    Claim: "The Bureau of Alcohol, Firearms and Tobacco  has reclassified wet nitrocellulose as a high explosive."
                    Category: Guns
                    Claim: "There are more guns in this country than there are people."
                    Category: Guns
                    Claim: "An academic study cited by conservative news organizations and the Trump administration proved that Hillary Clinton received more than 800,000 non-citizen votes in the 2016 presidential election."
                    Category: Ballot Box
                    Claim: "Tens of thousands of fraudulent Clinton votes found in Ohio warehouse."
                    Category: Ballot Box
                    Claim: "Members of the alt-right group the National Policy Institute can be seen on video doing a Nazi-like salute as speaker Richard Spencer calls out \"Hail Trump, hail our people, hail victory!\"
                    Category: Conspiracy Theories
                    Claim: "Osama bin Laden is still alive living in the Bahamas"
                    Category: Conspiracy Theories
                    Claim: "The Environmental Protection Agency will allow new asbestos products to enter the market."
                    Category: Environment
                    Claim: "Not one penny of California's Prop 67 bag ban tax goes to the environment."
                    Category: Environment
                    Claim: "Ben Carson said that illegal immigrants who get caught voting should be stripped of citizenship."
                    Category: Quotes
                    Claim: "Robert Redford Says: Michelle And Barack Obama Should Get Five Years In Prison."
                    Category: Quotes
                    Claim: "Says a photo of a park covered in trash was left behind by "environmentalists" after an Earth Day celebration in California."
                    Category: Imagery
                    Claim: "A social media video appears to show Nancy Pelosi slurring her words during an interview at the Center for American Progress Ideas Conference."
                    Category: Imagery
                    Claim: "Millions of Muslims protested against the Islamic State in November 2016, but a "media blackout" prevented people from hearing about the event."
                    Claim: Religion
                    Claim: "one cannot import a Koran published in the Arabic language"
                    Claim: Religion
                    Claim: "Border agents discover body of 6-year-old girl: 'She was raped By 30 men'."
                    Category: Subjective
                    Claim: "The inauguration of Donald Trump "will be the first one that I miss since I've been in Congress."
                    Category: Subjective
                    Claim: "Colorado has offered free birth control for five years, leading to: a 40 percent drop in unintended pregnancy, a 42 percent drop in abortions, and millions of dollars in healthcare savings."
                    Category: Abortion
                    What category does the following claim belong to?
                    Claim:{}
                    If the claim doesn’t belong to any of the provided categories, it should be categorized as “Other”. If the claim doesn't have enough
                    context to be verified, it should be categorized as "Subjective".
                    Your answer should be just the found category and no other text.'''.format(claim)

    return prompt

func_prompts = [construct_prompt]

def extract_claim_date(claim_context, time_offset):
    res = re.findall(REGEX, claim_context)
    if res:
        month, day, year = res[0]
        if int(day) < 10:
            day = '0' + day
        date_str = "{}-{}-{}".format(year, MONTH_MAPPING[month], day)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # add offset to the date so that we can experiment with different time constraints
        new_date = date_obj + timedelta(days=time_offset)
        # format the new date as "YYYY-MM-DD"
        new_date_str = new_date.strftime("%Y-%m-%d")
        return new_date_str
    else:
        return None

# Generate sub-questions and apply filtering 
def format_response(res):
    for q in res.splitlines():
        is_added = True
        q = q.strip()
        subcategory = q
        if len(q) != 0:
            # Check if it is a justification line
            try:
                q = q.split("Justification: ")[1]
                is_justification = True
            except:
                is_justification = False
                q = q.split("Question: ")[1]
            # Remove quotation mark if there are any
            q = re.sub('"', '', q)
            # Remove question number if there are any
            if q.lower().startswith(NUMBERS):
                q = q[3:]
                if q[0]==" ":
                    q = q[1:]
    return subcategory

def count_matches(golden_categories, categorization):
    matches = 0
    golden_categories = golden_categories.values
    subcategories = ['Politics', 'Politicians', 'Guns', 'Quotes', 'Immigration', 'Conspiracy Theories', 'Imagery', 'Ballot Box', 'Religion', 'Environment', 'Abortion']
    for golden_category, category in zip(golden_categories, categorization):
        if golden_category not in subcategories:
            golden_category = 'Other'
        if golden_category == category:
            matches = matches + 1
    accuracy = matches / len(golden_categories)
    return accuracy


def main(args):
    df = pd.read_json(args.input_path, lines=True)
    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end
    if args.include_first_category:
        df['category'] = ['Politics'] * end
    llm = ChatOpenAI(temperature = 0.7, model = ENGINE, api_key = api_key, max_tokens = 128, max_retries = MAX_GPT_CALLS)
    subcategories = []
    categories = []
    start_time = time.time()
    for i in tqdm(range(start, end)):
        try:
            claim = df.iloc[i]['claim']
            prompt_params={'numbed_of_questions':MAX_NUM_QUESTIONS}
            response = promptLLM(llm, func_prompts, claim, start_time=start_time, prompt_params=prompt_params)
            print(response.content)
            subcategories.append(response.content)
        except Exception as e:
            print("error caught", e)
            print('i=', i)
    if args.include_subcategory:
        df['subcategory'] = subcategories
    if args.check_matches:
        accuracy = count_matches(df['subcategory'], subcategories)
        print('Accuracy: {}',accuracy)
    df.to_json(args.output_path, orient='records', lines=True)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--check_matches', type=int, default=0)
    parser.add_argument('--include_first_category', type=int, default=0)
    parser.add_argument('--include_subcategory', type=int, default=0)
    parser.add_argument('--time_offset', type=int, default=1, help="add an offest to the time at which the claim was made")
    args = parser.parse_args()
    main(args)
