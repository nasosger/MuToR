"""
============================================MATH500 evaluation=======================================================
This script can be used to evaluated the checkpoints obtained by finetuning the pretrained LLMs, 
with both Next-Token (SFT) and MuToR.

Since we used bidirectional attention among the prefix tokens, we utilize the BiGemmaForCausalLM/BiLlamaForCausalLM
modules for inference.
These modules enables bidirectional attention in the prefill step, when the model processes the prefix.
Then they allow for standard autoregressive generation of the answer tokens, leveraging the KV cache.

To run this script:
- set load_from_Hub=True in case you load our pretrained checkpoints.
- set checkpoint_dir to the directory where the checkpoints were saved.
- set MODEL_ARCH -> determine the model's architecture.
- set MAX_NEW_TOKENS -> 768 (increasing to 2048 only yields negligible improvement.)

!! Before running, make sure you download the test set from https://github.com/openai/prm800k/blob/main/prm800k/math_splits/test.jsonl
and store it locally.!!
========================================================================================================================
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer
from models.bi_gemma import BiGemmaForCausalLM
from models.bi_llama import BiLlamaForCausalLM
import json
from typing import Dict, List
import numpy as np
import natsort
import gzip
from math_utils import *
import blobfile as bf
from generation_utils import generate_greedy_with_kv_cache

DEVICE="cuda"
MAX_NEW_TOKENS=748
MODEL_ARCH = "Llama"                         # choices: {"Gemma", "Llama"}
save_results=True
load_from_hub=False

checkpoint_dir = "/path/to/saved/checkpoints"
accuracies_save_path = os.path.join("/path/to/store/results", "run_id")             # fill the run_id to identify the experiment.

if load_from_hub:
    checkpoints=[checkpoint_dir]
    tokenizer_path = os.path.join(checkpoint_dir)
else:
    # given that the checkpoints were saved using our finetuning script.
    checkpoints = natsort.natsorted(os.listdir(checkpoint_dir))[:-1]
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer")

# The prompt that we used for downstream finetuning (adopted from https://github.com/dkopi/bitune)
template = lambda x: f"Question: {x}\n\nAnswer:"


# https://github.com/openai/prm800k/blob/main/prm800k/eval/eval.py
def json_loads(s: str) -> Dict:
    try:
        return orjson.loads(s)
    except Exception:
        return json.loads(s)    # fallback


def open_jsonl(file: str):
    if file.endswith(".gz"):
        return gzip.open(bf.BlobFile(file, "rb"))
    return bf.BlobFile(file, "r")


def _read_jsonl(file: str) -> List[Dict]:
    assert bf.exists(file), file
    with open_jsonl(file) as f:
        return [json_loads(l) for l in f.readlines() if l]


samples_path = "/local/path/to/MATH500/test/set"
print(f"Reading {samples_path}, this may take a while...")
samples = _read_jsonl(samples_path)
val_data = samples

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def evaluate_ckpt(model, tokenizer, val_data, template):
    idx = 0
    correct = 0
    dataset_size = len(val_data)
    
    model.eval()                                # set into eval mode...
    for task_id, problem in enumerate(val_data):
        print(f"task_id {task_id}")
        
        prompt = template(problem['problem'])
        input_ids = tokenizer.encode(prompt, return_tensors="pt") 

        print("type and level: ", problem["subject"], problem["level"])
        print(tokenizer.decode(input_ids[0], skip_special_tokens=False))

        if MODEL_ARCH == "Gemma":
            outputs = model.generate(
                                    input_ids.to(DEVICE), 
                                    max_new_tokens=MAX_NEW_TOKENS, 
                                    use_cache=True,
                                )
        elif MODEL_ARCH == "Llama":
            outputs = generate_greedy_with_kv_cache(
                                                    model, 
                                                    input_ids.to(DEVICE),
                                                    MAX_NEW_TOKENS,
                                                )
        decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens = False)
        print(decoded_outputs)
        
        output_str = last_boxed_only_string(decoded_outputs)
        answer_str = last_boxed_only_string(problem['solution'])

        # parse the answers
        output, answer = remove_boxed(output_str), remove_boxed(answer_str)
        print("Model's response:", output)
        print("Ground truth answer:", answer)
        equiv = is_equiv(output, answer)
        if equiv:
            correct += 1
        
        print(f"Correct: {correct} out of {idx+1}")
        print("=" * 40)
        idx += 1

    accuracy = correct/dataset_size
    return accuracy


accuracies = np.array([])

print("Checkpoints_to_be_evaluated: ", checkpoints)
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
    
    accuracy= evaluate_ckpt(model, tokenizer, val_data, template)
    accuracies = np.append(accuracies, accuracy)
    
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Final Accuracy: {accuracy}")
    del model


print("Model evaluated: ", checkpoint_dir)
print("Accuracies saved: ", accuracies)
print("Maximum response length", MAX_NEW_TOKENS)

if save_results:
    print("saving...")
    np.save(accuracies_save_path, accuracies)