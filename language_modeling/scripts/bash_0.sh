HOME_DIR="/path/to/home/dir"                                                # your/path/to/this/dir

run_id="run_id"                                                             # experiment id
wandb=1                                                                     # default = 1 (whether to use wandb)
save_dir="/dir/to/save/checkpoints/${run_id}/"                              # where to save training checkpoints
seed=42                                                                     # {42, 43, 44}

# MuToR's configuration
min_offset=0                                        # equivalent to d_min=1 in the paper
max_offset=3                                        # consider max_offset = d_max-1 (so for d_max=4 -> max_offset=3)
aux_loss_coeff=0.3

compile=0

batch_size=1                                    # Total_batch_size = 1 * gradient_accumulation_steps
lr_scheduler_type="linear"                      # lr scheduler -- default = "linear"
gradient_accumulation_steps=10                  # gradient accumulation steps -- 10 for Gemma, 2 for Llama.
seq_length=512                                  # max_sequence_length to keep
num_epochs=5                                    # epochs

weight_decay=0.0
beta1=0.9
beta2=0.999