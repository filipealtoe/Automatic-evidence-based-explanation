import argparse
import os
import re
import openai
import logging
import time
import json
import pandas as pd
from tqdm import tqdm

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")


# OpenAI Engine, feel free to change
ENGINE = 'gpt-4o'
# The stop criteria for question generation, feel free to change
MAX_GPT_RETRIES = 5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("file_management.log"), logging.StreamHandler()]
)

def construc_intermediate_prompt(part_number, total_parts=None, context_text=""):
    #According to https://medium.com/@josediazmoreno/break-the-limits-send-large-text-blocks-to-chatgpt-with-ease-6824b86d3270
    if part_number == 0:
        prompt = '''The total length of the content that I want to send you is too large to send in only one piece. 
        For sending the content, I will follow this rule: [START PART 1/''' + str(total_parts) + "] content [END PART 1/" + str(total_parts) + "]. You just answer: Received part 1/" + str(total_parts) + ''']. When I tell you "ALL PARTS SENT", you merge all parts and answer: "Parts concatenated".'''

    elif part_number == total_parts + 1:
        prompt = '''ALL PARTS SENT. '''

    else:
        prompt = '''Do not answer yet. This is just another part of the text I want to send you. Just receive, acknowledge as Part Received and wait for instructions.''' + "[START PART " + str(part_number) + "/" + str(total_parts) + "]" + context_text + "[END PART " + str(part_number) + "/" + str(total_parts) + "]"

        #prompt.format(str(part_number),str(total_parts),context_text,str(part_number),str(total_parts))
    return prompt

def construct_static_examples():
    showcase_examples = '''Hypothesis: "{}" The attached file is a scraped web page that may or may not provide evidence to the hypothesis. Summarize this text including only the passages that are relevant to confirm or deny the hypothesis. Output only the summary '''
    return showcase_examples

def construct_static_prompt():
    static_prompt = '''Hypothesis: "{}" Merge the previously sent Parts into a single context. Summarize the merged text including only the passages that are directlty relevant to the provided hypothesis. Output only the generated summary '''
    return static_prompt

# Format prompt for LLM call
def construct_prompt(showcase_examples, justification):
    return showcase_examples.format(justification)


def process_long_prompt(file_path, scraped_text, overwrite = True, create_file = False, prompt_max_words=5000):

    if create_file:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Check if file exists and overwrite is not allowed
        if not overwrite and os.path.isfile(file_path):
            raise FileExistsError(f"The file '{file_path}' already exists and overwrite is set to False.")
        
        if file_path.split(".")[2] == 'txt':
            # Write the string to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(scraped_text)
        elif file_path.split(".")[2] == 'jsonl':
            data = {"prompt": scraped_text, "completion": "This is a scraped web page content"}
            jason_string = json.dumps(data)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(jason_string)
        return None
    else:
        # Create smaller chunks to be fed into prompt chaining
        words = scraped_text.split()
        total_words = len(words)
        
        # Create chunks
        chunks = []
        for i in range(0, total_words, prompt_max_words):
            chunk = " ".join(words[i:i + prompt_max_words])
            chunks.append(chunk)
        return chunks
    
    return total_words, chunks


def LLM_retry_on_rate_limit(api_call, max_retries=MAX_GPT_RETRIES, retry_delay=1):

    for attempt in range(max_retries):
        try:
            return api_call()
        except openai.error.RateLimitError as e:
            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
            logging.warning(f"Rate limit error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            logging.error(f"API call failed: {e}")
            break
    logging.error(f"All retry attempts failed after {max_retries} retries.")
    return None

def LLM_upload_file(file_path):

    def api_call():
        with open(file_path, 'rb') as file:
            response = openai.File.create(
                file=file,
                purpose='fine-tune'  # Adjust depending on your intended purpose
            )
            return response['id']
    
    try:
        file_id = LLM_retry_on_rate_limit(api_call)
        if file_id:
            logging.info(f"File '{file_path}' uploaded successfully. File ID: {file_id}")
        return file_id
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        return None

def LLM_prompt_with_file_reference(file_id, prompt, model=ENGINE, temperature=0.7, max_tokens=1024):
 
    def api_call():
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response['choices'][0]['message']['content'].strip()

    response = LLM_retry_on_rate_limit(api_call)
    if response:
        logging.info(f"Prompt with file ID '{file_id}' completed successfully.")
    return response

def LLM_delete_uploaded_file(file_id):

    def api_call():
        response = openai.File.delete(file_id)
        return response['deleted']
    
    deleted = LLM_retry_on_rate_limit(api_call)
    if deleted:
        logging.info(f"File with ID '{file_id}' deleted successfully.")
    else:
        logging.error(f"Failed to delete file with ID '{file_id}'.")
    return deleted

def LLM_send_prompt(prompt):
    try:
        #prompt = "Context Part " + str(iteration) + ": \"" + scraped_text_chunk + ".\" Respond with Received." 
        res = openai.ChatCompletion.create(
            model=ENGINE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        res = res['choices'][0]['message']['content'].strip()
        if ENGINE == "gpt-4o":
            delay = 20
        else: delay = 2
        time.sleep(delay)
    except Exception as e:
                logging.error(f"Context Passing partial prompt failed: {e}")
    return res

def summarize_justification(decomposed_justification, scraped_text_file_path, scraped_text_chunks):
    if scraped_text_chunks != None:
        #iteration = 1
        part_number = 0
        for scraped_text_chunk in scraped_text_chunks:
            #If first iteration, send instructions on how to handle partioned context
            if part_number == 0:
                prompt = construc_intermediate_prompt(part_number, len(scraped_text_chunks))
                response = LLM_send_prompt(prompt)
                part_number = part_number + 1
                prompt = construc_intermediate_prompt(part_number, len(scraped_text_chunks), scraped_text_chunk)
                response = LLM_send_prompt(prompt)
                part_number = part_number + 1
            else:
                prompt = construc_intermediate_prompt(part_number, len(scraped_text_chunks), scraped_text_chunk)
                response = LLM_send_prompt(prompt)
                part_number = part_number + 1
            

            #iteration = iteration + 1
        Completion_prompt = construc_intermediate_prompt(part_number, len(scraped_text_chunks))
        #response = LLM_send_prompt(prompt)
        static_prompt = construct_static_prompt()
        prompt = construct_prompt(static_prompt, decomposed_justification)
        prompt = Completion_prompt + prompt
        response = LLM_send_prompt(prompt)
    #If uses fine tunning
    else:
        showcase_examples = construct_static_examples()
        prompt = construct_prompt(showcase_examples, decomposed_justification)
        file_id = LLM_upload_file(scraped_text_file_path)
        response = LLM_prompt_with_file_reference(file_id, prompt)
        LLM_delete_uploaded_file(file_id)
    return response


def main(args):
    df = pd.read_json(args.input_path, lines=True)
    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end
    local_scraped_text_file_path = os.path.dirname(args.input_path) + "/" + "scraped_text.jsonl"

    for i in tqdm(range(start, end)):
        try:
            decomposed_search_hits = df.iloc[i]['decomposed_search_hits']
            for decomposed_search_hit in decomposed_search_hits:
                decomposed_justification = decomposed_search_hit['decomposed_justification']
                i = 0
                for page_info in decomposed_search_hit['pages_info']:
                    if page_info['scraped_text'] != "":
                        scraped_text_chunks = process_long_prompt(local_scraped_text_file_path, page_info['page_content'])
                        page_info['justification_summary'] = summarize_justification(decomposed_justification, local_scraped_text_file_path, scraped_text_chunks)            
                        if os.path.isfile(local_scraped_text_file_path):
                            os.remove(local_scraped_text_file_path)
                        decomposed_search_hit['pages_info'][i] = page_info.copy()
                    i = i + 1
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
