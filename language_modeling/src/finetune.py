# Adapted from https://github.com/dkopi/bitune/finetune.py

import os
import time
import math
import logging
from collections import deque
import numpy as np
import torch
import datasets
import transformers
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.optimization import AdamW, get_scheduler
import wandb

from dataloaders_mutor import create_dataloaders
from args_mutor import MuToRArguments, update_config
from models.gemma_mutor import MuToRGemmaForCausalLM
from models.llama_mutor import MuToRLlamaForCausalLM


def log_trainable_params(accelerator, model):
    """
    Helper function to count the total number of trainable parameters.
    """
    _trainable_params = 0
    if accelerator.is_main_process:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
        _trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            "trainable params:",
            _trainable_params,
            flush=True,
        )
    return _trainable_params


def get_grouped_params(
    model,
    args,
    no_decay=[
        "bias",
        "ln_1.weight",
        "ln_2.weight",
        "ln_f.weight",
        "norm",
    ]
):
    """
    Helper function to group parameters wrt weight decay.
    """
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)

    return [
        {
            "params": params_with_wd,
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {"params": params_without_wd, "weight_decay": 0.0, "lr": args.learning_rate}
    ]


def setup_logging(args, accelerator):
    """
    Helper function to setup the Logger and also initialize tracking with wandb for each run. 
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.wandb_project,
            vars(args),
            init_kwargs={
                "wandb": {
                    "mode": "online" if args.wandb == 1 else "disabled",
                    "name": args.wandb_name,
                }
            },
        )
        run_name = accelerator.trackers[0].run.name
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ""
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, run_name


def log_metrics(logger, step, metrics, to_print=True, accelerator=None):
    """
    Helper function to log training metrics, both on terminal and wandb.
    """
    if to_print:
        logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process: 
        accelerator.log(metrics, step)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def process_input_and_labels_ntp(batch, args):
    """
    Preprocessing for baseline SFT (Next-Token):
    1) Masks out the question tokens from the labels (loss calculation ignores them.)
    2) Constructs custom 4D attention mask to allow bidirectional attention in the prefix.

    Since we experiment on a single A100 GPU, we use batch_size=1.
    !!! If your resources can handle larger batch_sizes, this function may need some modifications !!!
    """
    input_ids = batch["input_ids"]
    batch_size = input_ids.shape[0]
    sequence_len = input_ids.shape[-1]

    assert batch_size == 1, "Current version does not handle batch size > 1, please make the necessary modifications for padding."

    # Mask out padding tokens (if any) as well as prefix tokens from the loss computation.
    mask = batch["attention_mask"] == 1
    range_matrix = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(
        input_ids.shape[0], 1
    )
    _mask = range_matrix >= batch["question_sizes"].unsqueeze(1)
    mask = mask & _mask
    labels = torch.where(mask, input_ids, -100)

    # Construct the custom attention mask (additive), allowing bidirectional attention in the prefix tokens.
    attention_mask_4d = torch.zeros(batch_size, 1, sequence_len, sequence_len, dtype=torch.bfloat16)
    fill_inf = torch.finfo(torch.bfloat16).min

    for i in range(batch_size):
        question_length = batch["question_sizes"][i].item()
        dim_aux = sequence_len - question_length
    
        # causal mask for the answer part.
        mask_auxiliary = torch.full([dim_aux,dim_aux], fill_value= fill_inf, dtype=torch.bfloat16)           
        triu_mask = torch.triu(mask_auxiliary, diagonal=1)
        attention_mask_4d[i, 0, question_length:, question_length:] = triu_mask
        # prevent the question tokens from attending to the answer tokens.
        attention_mask_4d[i, 0, 0:question_length, question_length: ] = fill_inf        

    attention_mask_4d = attention_mask_4d.to(input_ids.device)

    return input_ids, labels, attention_mask_4d, batch["question_sizes"]


def process_input_and_labels_mutor(batch, args, reg_token_id, pad_token_id):
    """ 
    --- Preprocessing for MuToR --- 
    1) Interleave the register tokens in the answer sequence.
    2) Compute the labels tensor (as well as the modified position ids for the registers).
    3) Construct our custom attention mask, as defined in the paper.

    Since we experiment on a single A100 GPU, we use batch_size=1, so we do not need padding.
    !!! If your resources can handle larger batch_sizes, this function needs to be modified to handle padding as well. !!!
    """
    input_ids = batch["input_ids"]
    batch_size = input_ids.shape[0]
    sequence_len  = input_ids.shape[-1]

    assert batch_size == 1, "Current version does not handle batch size > 1, please make the necessary modifications for padding."

    # To align with our paper, offset = d - 1 ( max_offset = d_max - 1 )
    offset = np.random.randint(args.min_offset, args.max_offset + 1)           

    modified_input_ids = []
    attention_masks = []
    reg_token_indices_batch = []            # List of Tensors
    real_token_indices_batch = []           # List of Tensors
    
    for i in range(batch_size):             # batch_size=1 
        question_length = batch["question_sizes"][i].item()
        question_part_tensor = input_ids[i, :question_length]
        answer_part_tensor = input_ids[i, question_length:]

        # mask out the padding tokens if you need to handle padding as well -- might need modification for LMs when pad_token = eos
        # non_padding_mask = answer_part_tensor != pad_token_id
        # non_padding_answer = answer_part_tensor[non_padding_mask]

        reg_tokens = torch.full_like(answer_part_tensor, reg_token_id)
        interleaved_answer = torch.stack([reg_tokens, answer_part_tensor], dim=1).flatten(0)

        modified_sequence = torch.cat([question_part_tensor, interleaved_answer])
        attention = torch.ones_like(modified_sequence)

        reg_token_indices = torch.arange(0, len(interleaved_answer), 2, device=input_ids.device) + question_length
        total_len = modified_sequence.size(0)
        all_indices = torch.arange(total_len, device=input_ids.device)
        real_token_mask = torch.ones(total_len, dtype=torch.bool, device=input_ids.device)
        real_token_mask[reg_token_indices] = False
        real_token_indices = all_indices[real_token_mask]

        modified_input_ids.append(modified_sequence)
        attention_masks.append(attention)
        reg_token_indices_batch.append(reg_token_indices)
        real_token_indices_batch.append(real_token_indices)


    # update the batch - in case of batch size > 1, you need to pad first.
    batch["input_ids"] = torch.stack(modified_input_ids)                                    # [B, new_seq_len]
    batch["attention_mask"] = torch.stack(attention_masks)                                  # [B, new_seq_len]
    input_ids = batch["input_ids"]                          # [B, new_seq_len]
    sequence_len = input_ids.shape[-1]

    # create the labels  tensor and mask the question tokens (to ignore them during loss computation).
    mask = batch["attention_mask"] == 1
    range_matrix = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(input_ids.shape[0], 1)
    _mask = range_matrix >= batch["question_sizes"].unsqueeze(1)
    mask = mask & _mask
    labels = torch.where(mask, input_ids, -100)
    
    reg_position_ids_batch=[]
    for i in range(batch_size):
        question_length = batch["question_sizes"][i].item()
        reg_indices = reg_token_indices_batch[i]                            

        # Compute target indices and reg_positions as tensors
        j_tensor = torch.arange(len(reg_indices), device=input_ids.device)
        target_indices = reg_indices + 1 + 2 * offset                       # [num_regs]
        reg_positions = question_length + j_tensor + offset - 1             # [num_regs]

        # Mask: target index must be in bounds
        valid_targets_mask = target_indices < sequence_len

        # Assign labels only at valid positions
        labels[i, reg_indices[valid_targets_mask]] = input_ids[i, target_indices[valid_targets_mask]]
        labels[i, reg_indices[~valid_targets_mask]] = -100

        # For reg_position_ids, use calculated positions or clip to the last_valid_pos.
        last_valid_pos = len(real_token_indices_batch[i]) - 1
        reg_position_ids = torch.where(
            valid_targets_mask,
            reg_positions,
            torch.full_like(reg_positions, last_valid_pos)
        )

        reg_position_ids_batch.append(reg_position_ids)

    # Now construct the custom attention mask.
    # 4D shape: [batch_size, num_heads, sequence_len, sequence_len]       
    attention_mask_4d = torch.zeros(batch_size, 1, sequence_len, sequence_len, dtype=torch.bfloat16)
    fill_inf = torch.finfo(torch.bfloat16).min

    for i in range(batch_size):
        question_length = batch["question_sizes"][i].item()
        dim_aux = sequence_len - question_length
        # causal masking for the answer tokens
        mask_auxiliary = torch.full([dim_aux,dim_aux], fill_value= fill_inf, dtype=torch.bfloat16)           
        triu_mask = torch.triu(mask_auxiliary, diagonal=1)
        attention_mask_4d[i, 0, question_length:, question_length:] = triu_mask
        # prevent the question tokens from attending to the answer tokens      
        attention_mask_4d[i, 0, 0:question_length, question_length: ] = fill_inf     

        reg_indices = reg_token_indices_batch[i]
        real_token_indices = real_token_indices_batch[i]            

        # register tokens attend only to themselves and previous real tokens, not other register tokens.
        reg_i = reg_indices.unsqueeze(1)                            # [num_regs, 1]
        reg_j = reg_indices.unsqueeze(0)                            # [1, num_regs]
        reg_mask = torch.ones((len(reg_indices), len(reg_indices)), device=input_ids.device, dtype=torch.bool)
        reg_mask.fill_diagonal_(False)              # the registers should attend to themselves.

        # Apply masking only where reg_mask == True
        i_indices = reg_i.expand(-1, len(reg_indices))[reg_mask]
        j_indices = reg_j.expand(len(reg_indices), -1)[reg_mask]
            
        attention_mask_4d[i, 0, i_indices, j_indices] = fill_inf

        # prevent real tokens from attending to previous register tokens.
        real_i = real_token_indices.unsqueeze(1)                   # [num_real_tokens, 1]
        attention_mask_4d[i, 0, real_i, reg_indices] = fill_inf   
    
    attention_mask_4d = attention_mask_4d.to(input_ids.device)
    
    return input_ids, \
            labels, \
            attention_mask_4d, \
            reg_token_indices_batch, \
            real_token_indices_batch, \
            reg_position_ids_batch, \
            offset, \
            batch["question_sizes"]


def main():
    job_id = os.getenv("RUN_ID", str(int(time.time())))
    parser = HfArgumentParser(MuToRArguments)
    args = parser.parse_args()
    if os.getenv("WANDB") == "0":
        args.wandb = 0

    
    if args.seed is not None:
        set_seed(args.seed)

    if args.fsdp_distributed == 1:
        # In distributed scenarios (multi-GPU), set_seed was not affecting the dataloader in transformers version 4.43.0.
        # A workaround was to use a DataLoaderConfiguration with seedable sampler as below.
        # this fix was indeed solved in the latest versions of HF transformers (see: https://github.com/huggingface/accelerate/pull/3459 )
        # so probably by the time the code is uploaded, you can use the merged fix.
        dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
        accelerator = Accelerator(dataloader_config=dataloader_config, log_with=["wandb"])
    else:
        accelerator = Accelerator(log_with=["wandb"])

    samples_per_step = accelerator.state.num_processes * args.batch_size
    accelerator.print("samples per step: ", samples_per_step)

    logger, run_name = setup_logging(args, accelerator=accelerator)
    logger.info(accelerator.state)

    if accelerator.is_main_process:
        wandb.config.update({"job_id": job_id})

    model_config = AutoConfig.from_pretrained(args.model_hf)
    update_config(model_config, args)

    # add any other argument
    _kwargs = {}
    _kwargs["torch_dtype"]=torch.bfloat16

    if args.method_name=="Next_Token":
        model = AutoModelForCausalLM.from_pretrained(args.model_hf, **_kwargs)
    elif args.method_name=="MuToR":           
        if args.model_hf == "google/gemma-2b":
            model = MuToRGemmaForCausalLM.from_pretrained(args.model_hf, **_kwargs)
        elif args.model_hf == "meta-llama/Meta-Llama-3-8B":
            model = MuToRLlamaForCausalLM.from_pretrained(args.model_hf, **_kwargs)
        else:
            raise ValueError("Current implementation for MuToR includes Gemma 2B and Llama 3 8B")
    else: 
        raise ValueError("This repository supports only Next_Token (SFT) and MuToR.")

    # load pretrained tokenizer.   
    tokenizer = AutoTokenizer.from_pretrained(args.model_hf)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.method_name == "MuToR":
        # Add the "<reg>" token to the tokenizer, as a special token.
        special_tokens_dict = {'additional_special_tokens': ['<reg>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))       # Resize the embedding layer to initialize the register embedding.

    model.config.architectures[0] = model.__class__.__name__
    _trainable_params = log_trainable_params(accelerator, model)
    
    t_start_0 = time.time()   
    
    set_seed(args.seed)                     # so that we have the same shuffling in both NTP and MuToR      
    train_dataloader = create_dataloaders(tokenizer, args, accelerator)   
    # save_steps = (args.max_train_steps * 10) // args.num_ckpts_saved
    print(f"Creating dataloader took {time.time() - t_start_0} seconds", flush=True)

    optimizer = AdamW(get_grouped_params(model, args), betas=(args.beta1, args.beta2))
    # accelerator.print(optimizer)

    if accelerator.num_processes > 1:
        # calculate the max_train_steps for this scenario, to properly create the scheduler.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)
        args.max_train_steps = num_update_steps_per_epoch * args.num_epochs

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps if accelerator.num_processes == 1 else args.max_train_steps / 10,
        num_training_steps=args.max_train_steps,
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    if accelerator.num_processes > 1:                           # in case of distributed training, prepare the lr_scheduler as well.   
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    else:                                                       # in single-gpu, no need to prepare the lr_scheduler.
        model , optimizer, train_dataloader = accelerator.prepare(model , optimizer, train_dataloader)

    if accelerator.num_processes > 1:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) 
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    model.train()

    if args.compile:
        print("Compiling the model...")
        model.compile()

    completed_steps = 0
    t_start = time.time()
    t_start_0 = t_start
    loss_tracking = 0
    aux_loss_tracking = 0
    final_loss_tracking = 0
    running_loss_window=10
    running_loss = deque(maxlen=running_loss_window)
    running_aux_loss = deque(maxlen=running_loss_window)
    running_final_loss = deque(maxlen=running_loss_window)
    
    running_aux_loss_per_offset = []
    for i in range(args.min_offset, args.max_offset+1):
        running_aux_loss_per_offset.append(deque(maxlen=running_loss_window))

    reg_id = tokenizer.encode("<reg>", add_special_tokens=False)[0]         # Get the ID of the <reg> token
    pad_token_id = tokenizer.pad_token_id
     
    os.makedirs(args.save_dir, exist_ok=True)                               # Create the directory to save the checkpoints.

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total warmup steps = {args.num_warmup_steps}")

    accelerator.wait_for_everyone()
    step = 1
    _epoch = 1

    # Training loop
    while True:      
        if _epoch > args.num_epochs:                    # stop training
            break
        
        for batch in train_dataloader:

            if step < 5:                                # print the first batches.
                accelerator.print(batch, flush=True)
            
            if args.method_name == "MuToR":

                input_ids, \
                labels, \
                attention_mask, \
                reg_token_indices_batch, \
                real_token_indices_batch, \
                reg_position_ids_batch, \
                offset, \
                question_lengths = process_input_and_labels_mutor(batch, args, reg_id, pad_token_id)
                
                
                _, clm_loss, aux_loss = model(  input_ids, 
                                                labels=labels, 
                                                use_cache=False, 
                                                attention_mask = attention_mask, 
                                                real_token_indices=real_token_indices_batch, 
                                                reg_token_indices= reg_token_indices_batch, 
                                                register_position_ids=reg_position_ids_batch,
                                                question_lengths = question_lengths,
                                            )
            else:                                                   # Next-Token (baseline SFT)
                
                input_ids, \
                labels, \
                attention_mask, \
                question_lengths = process_input_and_labels_ntp(batch, args)

                out = model(input_ids, 
                            labels=labels, 
                            use_cache=False, 
                            attention_mask = attention_mask,
                        )
                clm_loss = out.loss

            avg_loss = (
                accelerator.gather(clm_loss.repeat(args.batch_size)).mean().item()
                / args.gradient_accumulation_steps
            )
            loss_tracking += avg_loss
            running_loss.append(avg_loss)
            
            clm_loss_divided = clm_loss / args.gradient_accumulation_steps          # divide by gradient_accum_steps

            if args.method_name=="MuToR":
                avg_aux_loss = (
                    accelerator.gather(aux_loss.repeat(args.batch_size)).mean().item()
                    / args.gradient_accumulation_steps
                )
                aux_loss_tracking += avg_aux_loss
                running_aux_loss.append(avg_aux_loss)
                aux_loss_divided = aux_loss / args.gradient_accumulation_steps
            
                final_loss = (1 - args.aux_loss_coeff) * clm_loss_divided + args.aux_loss_coeff * aux_loss_divided
            else:
                final_loss = clm_loss_divided
            
            final_avg_loss = (
                accelerator.gather(final_loss.repeat(args.batch_size)).mean().item()
            )
            final_loss_tracking += final_avg_loss
            running_final_loss.append(final_avg_loss)

            # log to wandb.
            metrics = {
                "samples": step * samples_per_step,
                "epoch": _epoch,
                "running_loss/clm_loss": sum(running_loss) / len(running_loss),
                "running_loss/aux_loss": sum(running_aux_loss) / len(running_aux_loss) if args.method_name=="MuToR" else 0,
                "loss_per_step/clm_loss": clm_loss_divided.item(),
                "loss_per_step/final_loss": final_loss.item(),
                "loss_per_step/aux_loss": aux_loss_divided.item() if args.method_name=="MuToR" else 0,
                "running_loss/final_loss": sum(running_final_loss) / len(running_final_loss),
                "steps": completed_steps,
                "trainable_params": _trainable_params,
                "offset": offset if args.method_name=="MuToR" else 0,
            }

            if args.method_name == "MuToR" and args.min_offset == 0:
                running_aux_loss_per_offset[offset].append(avg_aux_loss)
                metrics[f"running_loss/reg_loss_per_d_{offset+1}"] = sum(running_aux_loss_per_offset[offset]) / len(running_aux_loss_per_offset[offset])
                metrics[f"loss_per_step/reg_loss_per_d_{offset+1}"] = aux_loss_divided.item()

            if step % args.gradient_accumulation_steps != 0:
                accelerator.backward(final_loss)
            else:
                lr = get_lr(optimizer)
                accelerator.backward(final_loss)
    
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                elapsed_time = time.time() - t_start
                t_start = time.time()
                loss_tracking = 0
                completed_steps += 1

                metrics["grad_norm"] = grad_norm
                metrics["lr"] = lr
                metrics["time_per_iteration"] = elapsed_time
            
            step += 1
            log_metrics(
                logger,
                step,
                metrics,
                to_print=(step - 1) % args.gradient_accumulation_steps == 0,                      # set to True to log metrics for every steps.
                accelerator = accelerator
            )
            
            
            # you can use this to save evenly spaced checkpoints through training (by determining args.num_checkpoints_saved).
            # enable it for exact reproduction of our results when using 1M_GSM, 1M_MATH, 2M_GSM training splits.
            """
            if args.dataset=="nvidia/OpenMathInstruct-2" and args.fsdp_distributed == 0 and (step-1) % save_steps == 0:
                print(f"Saving after {step-1} number of steps...")
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_name = "ckpt_" + str(step-1)
                    save_dir_epoch = os.path.join(args.save_dir, save_name)
                    os.makedirs(save_dir_epoch, exist_ok=True)

                    unwrapped_model.save_pretrained(save_dir_epoch, save_function=accelerator.save)
                    print(f"Model saved at {save_dir_epoch}")
            """

        accelerator.wait_for_everyone()
            
        if accelerator.distributed_type == DistributedType.FSDP:
            save_name = "epoch_" + str(_epoch)
            save_dir_epoch = os.path.join(args.save_dir, save_name)
            os.makedirs(save_dir_epoch, exist_ok=True)

            accelerator.print("FSDP saving")
            start_saving_time = time.time()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                save_dir_epoch,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
            print(f"Process {accelerator.process_index} has reached this point.")
            accelerator.wait_for_everyone()
            accelerator.print(f"saving took {time.time() - start_saving_time}")
        else:
            # if args.dataset=="nvidia/OpenMathInstruct-2":
                # pass
            start_saving_time = time.time()
            save_name = "epoch_" + str(_epoch)
            save_dir_epoch = os.path.join(args.save_dir, save_name)
            os.makedirs(save_dir_epoch, exist_ok=True)
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                print("Single process saving.")
                unwrapped_model.save_pretrained(save_dir_epoch, save_function=accelerator.save)
            print(f"Process {accelerator.process_index} has reached this point.")
            accelerator.wait_for_everyone()
            accelerator.print(f"saving took {time.time() - start_saving_time}")
            accelerator.print(f"Model saved at {save_dir_epoch}")

            
        _epoch += 1

    _training_time = time.time() - t_start_0
    print(f"Training took {_training_time} seconds")

    # save the tokenizer.
    tokenizer_save_dir = os.path.join(args.save_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_save_dir)


if __name__ == "__main__":
    main()