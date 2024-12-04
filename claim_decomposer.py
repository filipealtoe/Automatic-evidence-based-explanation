import argparse
import os
import re
import openai
import pandas as pd
import time
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from LLMsummarizer import promptLLM

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

WH_MATCHES = ("why", "who", "which", "what", "where", "when", "how")

NUMBERS = ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")

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
    
    prompt = '''You are a fact-checker. Assume the following claim is false: "{}". 
    What would be the {} most important yes or no types of questions to be asked to verify the claim is false?
All questions need to have a no response if the claim is false. 
Provide only the questions and their corresponding justification without any other text in the following format: 
Question: 
Justification: .'''.format(claim, prompt_params['numbed_of_questions'])
    
    #prompt = '''You are a fact-checker. Assume the following claim is true: "{}". 
    #What would be the {} most important yes or no types of questions to be asked to verify the claim is true?
    #All questions need to have a yes response if the claim is true. 
    #Provide only the questions and their corresponding justification without any other text in the following format: 
    #Question: 
    #Justification: .'''.format(claim, prompt_params['numbed_of_questions'])

    return prompt

func_prompts = [construct_prompt]

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
                is_added = False

            if is_added and not is_justification:
                questions.append(q)
            elif is_added and is_justification:
                justifications.append(q)
    return questions, justifications


def main(args):
    df = pd.read_json(args.input_path, lines=True)

    # Add empty column if we don't have already
    df["claim questions"] = ""
    df["justifications"] = ""
    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end
    llm = ChatOpenAI(temperature = 0, model = ENGINE, api_key = api_key, max_tokens = 1024, max_retries = MAX_GPT_CALLS)
    start_time = time.time()
    for i in tqdm(range(start, end)):
        try:
            claim = df.iloc[i]['claim']
            llm_called = 0
            questions = []
            justifications = []
            response = promptLLM(llm, func_prompts, claim, start_time=start_time, prompt_params={'numbed_of_questions':MAX_NUM_QUESTIONS})
            questions, justifications = format_response(response.content, questions, justifications)
            df.at[i, 'claim questions'] = questions
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
    args = parser.parse_args()
    main(args)
