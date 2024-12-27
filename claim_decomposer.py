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


# Format prompt for GPT3 call
def construct_prompt(claim, prompt_params=None):
    #prompt_old = '''You are a fact-checker: What would be the {} most important yes or no types of questions to be asked to decompose the following claim: "{}"
#All questions need to have a yes response if the claim is true. 
#Provide only the questions and their corresponding justification without any other text in the following format: 
#Question: 
#Justification: .'''.format(prompt_params['numbed_of_questions'], claim)
    
    #prompt = '''You are a fact-checker. Assume the following claim is false: "{}". 
    #What would be the {} most important yes or no types of questions to be asked to verify the claim is false?
#All questions need to have a "no" response if the claim is false. 
#Provide only the questions and their corresponding justification without any other text in the following format: 
#Question: 
#Justification: .'''.format(claim, prompt_params['numbed_of_questions'])
    
    prompt = '''You are a fact-checker. Assume the following claim is true: "{}". 
    What would be the {} most important yes or no types of questions to be asked to verify the claim is true?
    All questions need to have a "yes" response if the claim is true. 
    Provide only the questions and their corresponding justification without any other text in the following format: 
    Question: 
    Justification: .'''.format(claim, prompt_params['numbed_of_questions'])

    #Improved Prompt using timeframe of claim
    prompt = ''' You are a fact-checker. A claim is true when the statement is accurate. A claim is false when the statement is not accurate.
    The following claim was published on {}: "{}"  
    Assume you will do a web search to verify the claim. What would be the {} most important yes or no questions to feed a web browser to verify if the claim is true? 
    All questions need to have a "yes" response if the claim is true and a "no" answer if the claim is false. 
    All the questions must explore different aspects of the claim.
    Return a single list of questions in the following format without any other text: 
    Question: 
    Justification:
    The top five to verify the claim is true and the bottom five to verify the claim is false.'''.format(prompt_params['claim_date'], claim, int(prompt_params['numbed_of_questions']))

    #Trying to not add bias towards true or false classification
    prompt = '''You are a fact-checker. A claim is true when the statement is accurate. A claim is false when the statement is not accurate.
    The following claim was published on {}: "{}" 
    Assume you will do a web search to verify the claim. What would be the {} most important yes or no types of questions to feed a web browser to verify the claim is true and the 
    {} most important yes or no types of questions to feed a web browser to verify the claim is false?
    The two sets of 5 questions must explore different aspects of the claim. 
    Return a single list of questions in the following format without any other text: 
    Question: 
    Justification:
    The top five to verify the claim is true and the bottom five to verify the claim is false.'''.format(prompt_params['claim_date'],claim, int(prompt_params['numbed_of_questions']/2), int(prompt_params['numbed_of_questions']/2))

    #Remove date since after improving the web search range filtering
    #Trying to not add bias towards true or false classification
    prompt = '''You are a fact-checker. A claim is true when the statement is accurate. A claim is false when the statement is not accurate.
    Take the following claim: "{}" 
    Assume you will do a web search to verify the claim. What would be the {} most important yes or no types of questions to feed a web browser to verify the claim is true and the 
    {} most important yes or no types of questions to feed a web browser to verify the claim is false?
    The two sets of 5 questions must explore different aspects of the claim. 
    Return a single list of questions in the following format without any other text: 
    Question: 
    Justification:
    The top five to verify the claim is true and the bottom five to verify the claim is false.'''.format(claim, int(prompt_params['numbed_of_questions']/2), int(prompt_params['numbed_of_questions']/2))

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
def format_response(res, questions, justifications):
    for q in res.splitlines():
        is_added = True
        q = q.strip()

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
            '''
            # Do not add question or justifications that has wh-word or is duplicate
            #if not q.isspace() and not any(x in q.lower() for x in WH_MATCHES):
            if not q.isspace():
                for q_alreay_in in questions:
                    # Duplicate if exact string match
                    if q == q_alreay_in:
                        is_added = False
                        break
                for q_alreay_in in justifications:
                    # Duplicate if exact string match
                    if q == q_alreay_in:
                        is_added = False
                        break
            else:
                is_added = False '''

            if is_added and not is_justification:
                questions.append(q)
            elif is_added and is_justification:
                justifications.append(q)
    return questions, justifications


def main(args):
    df = pd.read_json(args.input_path, lines=True)
    df["claim_questions"] = ""
    df["justifications"] = ""
    df["LLM_decomposing_prompt"] = ""
    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end
    llm = ChatOpenAI(temperature = 0.7, model = ENGINE, api_key = api_key, max_tokens = 1024, max_retries = MAX_GPT_CALLS)
    start_time = time.time()
    for i in tqdm(range(start, end)):
        try:
            claim = df.iloc[i]['claim']
            llm_called = 0
            questions = []
            justifications = []
            when_where = df.iloc[i]['venue']
            prompt_params={'numbed_of_questions':MAX_NUM_QUESTIONS, 'claim_date': extract_claim_date(when_where, args.time_offset)}
            df.at[i, 'LLM_decomposing_prompt'] =construct_prompt(claim, prompt_params)
            response = promptLLM(llm, func_prompts, claim, start_time=start_time, prompt_params=prompt_params)
            questions, justifications = format_response(response.content, questions, justifications)
            df.at[i, 'claim_questions'] = questions
            df.at[i, 'justifications'] = justifications
        except Exception as e:
            print("error caught", e)
            print('i=', i)

    df.to_json(args.output_path, orient='records', lines=True)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--time_offset', type=int, default=1, help="add an offest to the time at which the claim was made")
    args = parser.parse_args()
    main(args)
