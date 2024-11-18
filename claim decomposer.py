import argparse
import os
import re
import openai
import pandas as pd
from tqdm import tqdm

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

WH_MATCHES = ("why", "who", "which", "what", "where", "when", "how")

NUMBERS = ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")

# OpenAI Engine, feel free to change
ENGINE = 'gpt-4o'
# The stop criteria for question generation, feel free to change
MAX_GPT_CALLS = 5
MAX_NUM_QUESTIONS = 10


# Format example for static prompt
def construct_static_examples():
    showcase_examples = '''You are a fact-checker: What would be the ten most important yes or no types of questions to be asked to decompose the following claim: "{}"
Provide only the question and its corresponding justification, without any other text.'''
    return showcase_examples

# Format context for prompt
def construct_context(person, venue, claim):
    return "{} {} {}".format(
        person,
        venue,
        claim
    )


# Format prompt for GPT3 call
def construct_prompt(showcase_examples, claim):
    return showcase_examples.format(claim)


# Generate sub-questions and apply filtering 
def decompose_claim(claim, questions, justifications, simulate_LLM=1):
    showcase_examples = construct_static_examples()
    prompt = construct_prompt(showcase_examples, claim)
    if not simulate_LLM:
        res = openai.Completion.create(engine=ENGINE, prompt=prompt,
                                    temperature=0.7, max_tokens=1024,
                                    stop=["Claim"])
        res = res['choices'][0]["text"].strip()
    else:
        res = '''1. Has the US reduced its total carbon emissions over the last decade?
Justification: Establishes if emissions are actually decreasing in the US.

2. Are the per capita carbon emissions in the US lower today than they were a decade ago?
Justification: Measures reduction on a per-person basis, which reflects population-adjusted progress.

3. Is the reduction in US carbon emissions greater than in other industrialized nations?
Justification: Allows comparison with countries of similar economic status.

4. Have specific sectors in the US, like energy production, shown significant emission reductions?
Justification: Identifies if sectoral changes are driving the overall reduction.

5. Has the US implemented federal policies specifically aimed at reducing carbon emissions?
Justification: Determines if policy changes are actively contributing to emission reductions.

6. Are there regions or states within the US with increased carbon emissions despite national trends?
Justification: Tests if reductions are uniform across the country or regionally concentrated.

7. Have US reductions in emissions been sustained during periods of economic growth?
Justification: Confirms if reductions are resilient to economic cycles, which often increase emissions.

8. Do other parts of the world have higher emissions reduction targets or commitments compared to the US?
Justification: Assesses if other regions have comparable or more ambitious goals for emission reduction.

9. Has the US decreased its use of fossil fuels, specifically coal, compared to other parts of the world?
Justification: Examines the decline of high-emission energy sources as a contributor to reductions.

10. Are US carbon emissions reductions offset by increases in consumption of goods produced in high-emission countries?
Justification: Verifies if emission reductions are genuine or result from shifting carbon-intensive production abroad.'''

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

    for i in tqdm(range(start, end)):
        try:
            #context = construct_context(
                #df.iloc[i]['person'], df.iloc[i]['venue'], df.iloc[i]['claim'])
            claim = df.iloc[i]['claim']
            gpt3_called = 0
            questions = []
            justifications = []
            while (len(questions) < MAX_NUM_QUESTIONS
                   and gpt3_called < MAX_GPT_CALLS):
                questions, justifications = decompose_claim(
                    claim, questions, justifications, args.simulate_llm
                )
                gpt3_called += 1

            df.at[i, 'claim questions'] = questions
            df.at[i, 'justifications'] = justifications
        except Exception as e:
            print("error caught", e)
            print('i=', i)

    df.to_json(args.output_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--simulate_llm', type=int, default=1)
    args = parser.parse_args()
    main(args)
