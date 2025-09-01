################
source scripts/bash_0.sh 
################

prefix=""
max_train_steps=3735                          # max_train_steps define the number of update steps
num_warmup_steps=374                          # 10% of the total steps
num_ckpts_saved=5                             # save 5 checkpoints (evenly spaced)

# dataset choices: {"gsm8k", "nvidia/OpenMathInstruct-2"}.                     
dataset="gsm8k"
math_split=0                                   # set to 1 to use our 1M-Math training split

# model choices: {"google/gemma-2b", "meta-llama/Meta-Llama-3-8B"}
model_hf="google/gemma-2b" 
fsdp_distributed=0                                # set to 1 for the 8B model to train with FSDP

# method_name: {"Next_Token", "MuToR"}
method_name="MuToR"

# peak lr: {"Gemma 2B": "5e-5", "Llama 3 8B": "2e-5"}
learning_rate="5e-5"

##################
source scripts/bash_1.sh
##################