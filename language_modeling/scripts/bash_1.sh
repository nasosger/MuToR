
suffix=""
suffix="${suffix}[d_max][$((max_offset + 1))]"
suffix="${suffix}[aux_loss_coeff][${aux_loss_coeff}]"

suffix="${suffix}[${run_id}][${seed}]"
wandb_name="${method_name}${suffix}"

CUDA_VISIBLE_DEVICES=0 HOME_DIR=$HOME_DIR RUN_ID=$run_id accelerate launch src/finetune.py \
    --lr_scheduler_type $lr_scheduler_type \
    --weight_decay $weight_decay \
    --model_hf $model_hf \
    --num_epochs $num_epochs \
    --dataset $dataset \
    --seq_length $seq_length \
    --num_warmup_steps $num_warmup_steps \
    --max_train_steps $max_train_steps \
    --learning_rate $learning_rate \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --seed $seed \
    --wandb_name "$wandb_name" \
    --method_name "$method_name" \
    --wandb $wandb \
    --save_dir $save_dir \
    --min_offset $min_offset \
    --max_offset $max_offset \
    --aux_loss_coeff $aux_loss_coeff \
    --num_ckpts_saved $num_ckpts_saved \
    --math_split $math_split \
    --fsdp_distributed $fsdp_distributed \
    --compile $compile