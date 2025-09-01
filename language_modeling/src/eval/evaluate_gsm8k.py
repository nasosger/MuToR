"""
============================================GSM8K evaluation=======================================================
This script can be used to evaluated the checkpoints obtained by finetuning the pretrained LLMs, 
with both Next-Token (SFT) and MuToR.

Since we used bidirectional attention among the prefix tokens, we utilize the BiGemmaForCausalLM/BiLlamaForCausalLM
modules for inference.
These modules enables bidirectional attention in the prefill step, when the model processes the prefix.
Then they allow for standard autoregressive generation of the answer tokens, leveraging the KV cache.

To run this script:
- set load_from_Hub=True in case you load our pretrained checkpoints.
- set checkpoint_dir: either 1) local the directory where the checkpoints were saved or 2) the Hugging Face model hub path.
- set MODEL_ARCH -> determine the model's architecture (Gemma or Llama).
- set the TRAINING_DATASET -> it will determine the answer parser.
- set MAX_NEW_TOKENS -> 512 is sufficient for these experiments.
========================================================================================================================
"""
import sys
import os
from transformers import AutoTokenizer
from datasets import load_dataset
import re
import numpy as np
import natsort
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.bi_gemma import BiGemmaForCausalLM
from models.bi_llama import BiLlamaForCausalLM
from generation_utils import generate_greedy_with_kv_cache


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
DEVICE = "cuda"      
MAX_NEW_TOKENS=512
save_results=True
load_from_hub=False

checkpoint_dir = "/path/to/saved/checkpoints"                                       # local or HF model hub path.
accuracies_save_path = os.path.join("/path/to/store/results", "run_id")             # fill the run_id to identify the experiment.
TRAINING_DATASET = "gsm8k"                   # choices: {"gsm8k", "OpenMath"}
MODEL_ARCH = "Gemma"                         # choices: {"Gemma", "Llama"}

if load_from_hub:
    checkpoints=[checkpoint_dir]
    tokenizer_path = os.path.join(checkpoint_dir)
else:
    # given that the checkpoints were saved using our finetuning script.
    checkpoints = natsort.natsorted(os.listdir(checkpoint_dir))[:-1]
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer")

# The prompt that we used for downstream finetuning (adopted from https://github.com/dkopi/bitune)
template = lambda x: f"Question: {x}\n\nAnswer:"


def extract_answer(completion):
    """
    Answer parser function (https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py)
    """
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def extract_boxed_answer(completion):
    """
    Answer parser when the answer is enclosed in `\boxed{}` (OpenMath).
    """
    match = re.search(r"\\boxed\{(.*?)\}", completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def maybe_remove_comma(x: str) -> str:
  # Example: 5,600 -> 5600
  return x.replace(',', '')


# load the data
dataset = load_dataset("gsm8k", "main")
val_data = dataset['test']
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def evaluate_ckpt(model, tokenizer, val_data, template):
    all_responses = {}
    short_responses = {}
    idx = 0
    correct = 0
    dataset_size = len(val_data['question'])

    model.eval()
    for task_id, problem in enumerate(val_data):
        if task_id in all_responses: continue
        print(f"task_id {task_id}")
        
        prompt = template(problem['question'])
        input_ids = tokenizer.encode(prompt, return_tensors="pt") 
        print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
        
        if MODEL_ARCH == "Gemma":
            outputs = model.generate(input_ids.to(DEVICE), 
                                    max_new_tokens=MAX_NEW_TOKENS, 
                                    use_cache=True)
        elif MODEL_ARCH == "Llama":
            # we utilized a custom generation loop for Llama due to some warnings in the HF one.
            outputs = generate_greedy_with_kv_cache(
                                            model, 
                                            input_ids.to(DEVICE),
                                            MAX_NEW_TOKENS
                                        )
        
        decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens = False)
        print("Model's response: ", decoded_outputs)

        gt = extract_answer(problem['answer'])                      # extract the ground truth by gsm8k parser
        # extract the answer using the correct parser.
        if TRAINING_DATASET == "gsm8k":
            extracted_answer = extract_answer(decoded_outputs)
        else:
            extracted_answer = extract_boxed_answer(decoded_outputs)
        
        all_responses[task_id] = decoded_outputs
        short_responses[task_id] = extracted_answer
        print(f"Short answer: {short_responses[task_id]}")
        try:
            correct += float(maybe_remove_comma(gt)) == maybe_remove_comma(float(short_responses[task_id]))
        except:
            if (maybe_remove_comma(gt) == maybe_remove_comma(short_responses[task_id])):
                correct += 1

        print('-'*40)
        print(f"Short ground truth answer {gt}")
        print(f"Correct: {correct} out of {idx+1}")
        print("="*40)
        idx += 1

    accuracy = correct/dataset_size
    return accuracy


print("Checkpoints to be evaluated: ", checkpoints)
accuracies = np.array([])
for checkpoint in checkpoints:
    print(f"Evaluating checkpoint: {checkpoint}")
    
    if load_from_hub:
        checkpoint_path = checkpoint
    else:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    
    if MODEL_ARCH == "Gemma":
        model = BiGemmaForCausalLM.from_pretrained(checkpoint_path)
    elif MODEL_ARCH == "Llama":
        model = BiLlamaForCausalLM.from_pretrained(checkpoint_path)
    else:
        raise ValueError("Unsupported Language Model Architecture!")
    model.to(DEVICE)

    accuracy = evaluate_ckpt(model, tokenizer, val_data, template)
    accuracies = np.append(accuracies, accuracy)
    
    print(f"Checkpoint: {checkpoint}")
    print(f"Final Accuracy: {accuracy}") 
    del model


print("Model evaluated: ", checkpoint_dir)
print("Accuracies saved: ", accuracies)
print("Maximum response length: ", MAX_NEW_TOKENS)

if save_results:
    print("saving...")
    np.save(accuracies_save_path, accuracies)