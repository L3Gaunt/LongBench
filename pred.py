import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
from multiprocessing import Queue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

# Load and process the interjections prompt
interjections_prompt_template = open('prompts/interjections_prompt.txt', encoding='utf-8').read()

def process_interjections_prompt(question, choice_A, choice_B, choice_C, choice_D, answer):
    """Process the interjections prompt by replacing placeholders with actual values."""
    return interjections_prompt_template.replace('$Q$', question.strip())\
                                       .replace('$C_A$', choice_A.strip())\
                                       .replace('$C_B$', choice_B.strip())\
                                       .replace('$C_C$', choice_C.strip())\
                                       .replace('$C_D$', choice_D.strip())

def interject_document(document, interjection, tokenizer, frequency):
    """
    Interject the document with the interjection at specified token frequency.
    
    Args:
        document (str): The document to interject
        interjection (str): The interjection to insert
        tokenizer: The tokenizer to use for counting tokens
        frequency (int): The frequency (in tokens) at which to interject
        
    Returns:
        str: The document with interjections
    """
    if frequency <= 0:
        return document
        
    # Tokenize the document
    tokens = tokenizer.encode(document)
    
    # If document is shorter than frequency, return as is
    if len(tokens) <= frequency:
        return document
    
    # Split the tokens at frequency intervals
    chunks = [tokens[i:i+frequency] for i in range(0, len(tokens), frequency)]
    
    # Decode each chunk and join with the interjection
    result = ""
    for i, chunk in enumerate(chunks):
        chunk_text = tokenizer.decode(chunk)
        result += chunk_text
        if i < len(chunks) - 1:  # Don't add interjection after the last chunk
            result += interjection
    
    return result

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    # truncate
    truncated = False
    # max_len = maxlen_map[model]
    # if model in model_map:
    #     input_ids = tokenizer.encode(prompt)
    #     if len(input_ids) > max_len:
    #         input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
    #         prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    #         truncated = True
    # else:
    #     input_ids = tokenizer.encode(prompt, disallowed_special=())
    #     if len(input_ids) > max_len:
    #         input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
    #         prompt = tokenizer.decode(input_ids)
    #         truncated = True
    tries = 0
    
    # Remap model name to format needed by inference server if specified
    if model in model_map:
        model = model_map[model]
        
    while tries < 1: # only 1 try for now
        tries += 1
        try:
            messages = [{"role": "user", "content": prompt}]
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            choice = completion.choices[0].message.content
            print(*messages)
            print(*completion)
            return choice, truncated
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            #time.sleep(1) # don't sleep because we don't retry
    else:
        print("Max tries. Failed.")
        return '', truncated

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, args, result_queue):
    model = args.model
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer .from_pretrained(
            model_map[model], 
            trust_remote_code=True,
            token=HF_TOKEN
        )
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )
    for item in tqdm(data):
        context = item['context']
        
        # Process the interjections prompt if interjection_frequency > 0
        processed_interjection = ""
        if args.interjection_frequency > 0:
            processed_interjection = process_interjections_prompt(
                item['question'], 
                item['choice_A'], 
                item['choice_B'], 
                item['choice_C'], 
                item['choice_D'], 
                item['answer']
            )
        
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
            
        # Apply interjections to the context if interjection_frequency > 0
        if args.interjection_frequency > 0:
            context = interject_document(context, processed_interjection, tokenizer, args.interjection_frequency)
            
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        
        if args.cot:
            output, truncated_first = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=1024)
        else:
            output, truncated = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output, truncated_second = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128)
            if output == '':
                continue
            truncated = truncated_first or truncated_second  # Either prompt was truncated
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        item['truncated'] = truncated
        result_queue.put(item)

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    
    # Base filename without interjection info
    if args.rag > 0:
        base_filename = args.model.split("/")[-1] + f"_rag_{str(args.rag)}"
    elif args.no_context:
        base_filename = args.model.split("/")[-1] + "_no_context"
    elif args.cot:
        base_filename = args.model.split("/")[-1] + "_cot"
    else:
        base_filename = args.model.split("/")[-1]
    
    # Add interjection frequency to filename if applicable
    if args.interjection_frequency > 0:
        out_file = os.path.join(args.save_dir, base_filename + f"_if_{str(args.interjection_frequency)}.jsonl")
    else:
        out_file = os.path.join(args.save_dir, base_filename + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    # Create a queue for results
    result_queue = Queue()
    
    # Start processes
    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, result_queue))
        p.start()
        processes.append(p)

    # Write results to file
    with open(out_file, 'a', encoding='utf-8') as fout:
        while any(p.is_alive() for p in processes):
            try:
                item = result_queue.get(timeout=1)
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                fout.flush()
            except:
                continue

    # Wait for all processes to complete
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true')
    parser.add_argument("--no_context", "-nc", action='store_true')
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--interjection_frequency", "-if", type=int, default=0, 
                        help="Frequency (in tokens) at which to interject the interjections prompt. 0 means no interjections.")
    args = parser.parse_args()
    main()