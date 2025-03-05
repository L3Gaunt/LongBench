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

def interject_document(document, interjection, frequency, model=None, client=None):
    """
    Interject the document with the interjection at specified word frequency.
    Creates alternating user/system messages with real model completions.
    
    Args:
        document (str): The document to interject
        interjection (str): The interjection to insert
        frequency (int): The frequency (in words) at which to interject
        model (str): The model to use for generating completions
        client: The OpenAI client instance
        
    Returns:
        list: List of message dictionaries alternating between user and system roles
    """
    if frequency <= 0 or not model or not client:
        return [{"role": "user", "content": document}]
        
    # Split into words
    words = document.split()
    
    # If document is shorter than frequency, return as is
    if len(words) <= frequency:
        return [{"role": "user", "content": document}]
    
    # Split the words at frequency intervals
    chunks = [words[i:i+frequency] for i in range(0, len(words), frequency)]
    
    # Create alternating messages
    messages = []
    for i, chunk in enumerate(chunks):
        chunk_text = ' '.join(chunk)
        if i > 0:
            chunk_text = '<text>' + chunk_text
        
        messages.append({"role": "user", "content": chunk_text + "</text>" + interjection})
        if i < len(chunks) - 1:  # Don't add interjection after the last chunk
            # Get real completion for reading acknowledgment
            current_messages = messages.copy()
#            current_messages.append({"role": "user", "content": "Please acknowledge that you've read this part of the document and are ready to continue. Keep your response brief."})
            read_response = query_llm(current_messages, model, client, temperature=0.1, max_new_tokens=128)
            messages.append({"role": "assistant", "content": read_response})
                
    return messages

def query_llm(prompt, model, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    tries = 0
    
    # Remap model name to format needed by inference server if specified
    if model in model_map:
        model = model_map[model]
        
    while tries < 1: # only 1 try for now
        tries += 1
        try:
            # If prompt is already a list of messages, use it directly
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            choice = completion.choices[0].message.content
            print(*messages)
            print(*completion)
            return choice
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            return ''

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

def is_within_length_limit(text, model):
    """
    Check if the text length is within the model's token limit.
    Uses character count as a proxy with a 4x buffer.
    
    Args:
        text (str): The text to check
        model (str): The model name
        
    Returns:
        bool: True if the text is likely within limits, False otherwise
    """
    # Get the model's max token length, default to smallest if model not found
    max_tokens = maxlen_map.get(model, min(maxlen_map.values()))
    
    # Use character count as a proxy for token count
    # We use a 4x buffer as requested
    char_limit = max_tokens * 4
    
    return len(text) <= char_limit

class Statistics:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0
        self.error = 0
        self.skipped_length = 0
        self.total = 0
        
    def print_summary(self):
        print("\nProcessing Summary:")
        print(f"Total items: {self.total}")
        print(f"Correct: {self.correct} ({(self.correct/self.total*100):.2f}%)")
        print(f"Incorrect: {self.incorrect} ({(self.incorrect/self.total*100):.2f}%)")
        print(f"API Errors: {self.error} ({(self.error/self.total*100):.2f}%)")
        print(f"Skipped (too long): {self.skipped_length} ({(self.skipped_length/self.total*100):.2f}%)")
        print(f"Processed: {self.correct + self.incorrect} ({((self.correct + self.incorrect)/self.total*100):.2f}%)")
        
    def to_dict(self):
        total = self.total or 1  # Avoid division by zero
        return {
            "total_items": self.total,
            "correct": {
                "count": self.correct,
                "percentage": (self.correct/total*100)
            },
            "incorrect": {
                "count": self.incorrect,
                "percentage": (self.incorrect/total*100)
            },
            "api_errors": {
                "count": self.error,
                "percentage": (self.error/total*100)
            },
            "skipped_length": {
                "count": self.skipped_length,
                "percentage": (self.skipped_length/total*100)
            },
            "processed": {
                "count": self.correct + self.incorrect,
                "percentage": ((self.correct + self.incorrect)/total*100)
            }
        }

def get_pred(data, args, result_queue):
    model = args.model
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )
    
    stats = Statistics()
    stats.total = len(data)
    
    for item in tqdm(data):
        context = item['context']
        
        # Skip if context is too long
        if not is_within_length_limit(context, model):
            print(f"\nSkipped - Input too long: ID {item.get('_id', 'unknown')} ({len(context)} chars)")
            stats.skipped_length += 1
            continue
            
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
            
            # Check combined context length after RAG
            if not is_within_length_limit(context, model):
                print(f"\nSkipped - RAG context too long: ID {item.get('_id', 'unknown')} ({len(context)} chars)")
                stats.skipped_length += 1
                continue
                
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
            
        # Apply interjections to the context if interjection_frequency > 0
        if args.interjection_frequency > 0:
            messages = interject_document(context, processed_interjection, args.interjection_frequency, model, client)
            # Replace the context with the final message content
            context = ' '.join(msg["content"] for msg in messages)
            
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        
        try:
            if args.interjection_frequency > 0:
                # If we have interjections, add the final prompt as a user message
                messages.append({"role": "user", "content": prompt})
                output = query_llm(messages, model, client, temperature=0.1, max_new_tokens=1024 if args.cot else 128)
            else:
                output = query_llm(prompt, model, client, temperature=0.1, max_new_tokens=1024 if args.cot else 128)
                
            if output == '':
                print(f"\nAPI Error: ID {item.get('_id', 'unknown')}")
                stats.error += 1
                continue
                
            if args.cot: # extract answer
                response = output.strip()
                item['response_cot'] = response
                prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
                output = query_llm(prompt, model, client, temperature=0.1, max_new_tokens=128)
                if output == '':
                    print(f"\nAPI Error (COT): ID {item.get('_id', 'unknown')}")
                    stats.error += 1
                    continue
                    
            response = output.strip()
            item['response'] = response
            item['pred'] = extract_answer(response)
            item['judge'] = item['pred'] == item['answer']
            
            if item['judge']:
                stats.correct += 1
            else:
                stats.incorrect += 1
                
            item['context'] = context[:1000]
            result_queue.put(item)
            
        except Exception as e:
            print(f"\nUnexpected Error: ID {item.get('_id', 'unknown')} - {str(e)}")
            stats.error += 1
            continue
    
    # Send statistics through queue for the main process
    result_queue.put({"statistics": stats})

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
        base_filename += f"_if_{str(args.interjection_frequency)}"
    
    out_file = os.path.join(args.save_dir, base_filename + ".jsonl")
    summary_file = os.path.join(args.save_dir, base_filename + "_summary.json")

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

    # Collect statistics from all processes
    all_stats = []
    
    # Write results to file
    with open(out_file, 'a', encoding='utf-8') as fout:
        while any(p.is_alive() for p in processes) or not result_queue.empty():
            try:
                item = result_queue.get(timeout=1)
                if "statistics" in item:
                    all_stats.append(item["statistics"])
                else:
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                    fout.flush()
            except:
                continue

    # Wait for all processes to complete
    for p in processes:
        p.join()
        
    # Combine statistics from all processes and save summary
    if all_stats:
        combined_stats = Statistics()
        for stats in all_stats:
            combined_stats.total += stats.total
            combined_stats.correct += stats.correct
            combined_stats.incorrect += stats.incorrect
            combined_stats.error += stats.error
            combined_stats.skipped_length += stats.skipped_length
        
        # Print final summary
        print("\nFinal Statistics Across All Processes:")
        combined_stats.print_summary()
        
        # Save parameters and statistics to summary file
        summary = {
            "parameters": vars(args),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": combined_stats.to_dict()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--cot", "-cot", action='store_true')
    parser.add_argument("--no_context", "-nc", action='store_true')
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--interjection_frequency", "-if", type=int, default=0, 
                        help="Frequency (inverse - words per interjection) at which to interject the interjections prompt. 0 means no interjections.")
    args = parser.parse_args()
    main()