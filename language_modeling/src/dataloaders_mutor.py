import datasets
from torch.utils.data import DataLoader


def prepare_datasets(
    args,
    dataset_name,
    tokenizer,
    q_func,
    ans_func,
    train_field="train",
    val_field=None,
    subset=None,
    accelerator=None
):
    datasets.disable_caching()
    if args.dataset=="nvidia/OpenMathInstruct-2":
        # Load  OpenMathInstruct-2, specify the split. Filter accordingly.
        train_dataset = datasets.load_dataset(args.dataset, split="train_1M")
        # rename columns to align with gsm8k
        train_dataset = train_dataset.rename_column("problem", "question")
        train_dataset = train_dataset.rename_column("generated_solution", "answer")
        if args.math_split==0:                  # 1M-GSM or 2M-GSM
            train_dataset = train_dataset.filter(lambda example: example["problem_source"] == "augmented_gsm8k" or example["problem_source"] == "gsm8k")
        else:                                   # 1M-MATH
            train_dataset = train_dataset.filter(lambda example: example["problem_source"] == "augmented_math" or example["problem_source"] == "math")
            train_dataset = train_dataset.shuffle(seed=0)                   # set the seed for reproducible behaviour                    

            # keep the first 200K samples.
            train_dataset = train_dataset.select(range(200000))
    else:                   # on the other datasets.
        ds = datasets.load_dataset(dataset_name, subset)


    def tokenize(samples, args):
        """
        Tokenize the dataset.
        """
        samples = [dict(zip(samples, i)) for i in zip(*samples.values())]
        _questions = []
        _full = []

        for _, sample in enumerate(samples):
            
            q = q_func(sample)
            ans = ans_func(sample)
            # full = q + ans
            _questions.append(q)
            _full.append(q+ans)


        questions = tokenizer(
            _questions,
            padding="do_not_pad" if args.batch_size > 1 else "do_not_pad",    # if you need padding, use the DataCollator instead
            truncation=True if args.seq_length > 0 else False,
            max_length=args.seq_length if args.seq_length > 0 else None,
        )

        full_labels = tokenizer(
            _full,
            padding="do_not_pad" if args.batch_size > 1 else "do_not_pad",
            truncation=True if args.seq_length > 0 else False,
            max_length=args.seq_length if args.seq_length > 0 else None,
        )
        question_sizes = [
            len([_q for _q in q if _q != tokenizer.pad_token_id])
            for q in questions["input_ids"]
        ]
        full_sizes = [
            len([_f for _f in f if _f != tokenizer.pad_token_id])
            for f in full_labels["input_ids"]
        ]
        
        return {
            "input_ids": full_labels["input_ids"],
            "attention_mask": full_labels["attention_mask"],
            "question_sizes": question_sizes,
            "full_sizes":full_sizes,
        }
        
    
    if args.dataset=="nvidia/OpenMathInstruct-2":
        pass
    else:
        train_dataset = ds[train_field]
    
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            lambda samples: tokenize(samples, args),
            remove_columns=train_dataset.column_names,
            batched=True,
            num_proc=24,
        )
    with accelerator.main_process_first():
        train_dataset = train_dataset.filter(
            lambda samples: [
                ids[-1] == tokenizer.eos_token_id or ids[-1] == tokenizer.pad_token_id
                for ids in samples["input_ids"]
            ],
            batched=True,
            num_proc=24,
        )
    train_dataset = train_dataset.with_format("torch")
    valid_dataset  = None
    
    return train_dataset, valid_dataset

    


def create_dataloaders(tokenizer, args, accelerator):
    datasets.disable_caching()
    
    # locally save the dataset in case you want to skip preprocessing.
    _path = f"./cached_datasets/{args.model_hf}_{args.dataset}_{args.seq_length}_{args.batch_size}_{args.seed}"
    
    if args.dataset == "gsm8k":                                                   # Default templates
        train_dataset, _ = prepare_datasets(
            args,
            "gsm8k",
            tokenizer,
            lambda x: f"Question: {x['question']}\n\nAnswer:",
            lambda x: " " + x["answer"] + tokenizer.eos_token,
            subset="main",
            train_field="train",
            val_field="test",
            accelerator=accelerator
        )
    elif args.dataset == "nvidia/OpenMathInstruct-2":
        train_dataset, _ = prepare_datasets(
            args,
            "nvidia/OpenMathInstruct-2",
            tokenizer,
            lambda x: f"Question: {x['question']}\n\nAnswer:",
            lambda x: " " + x["answer"] + tokenizer.eos_token,
            train_field=None,
            accelerator=accelerator,
        )
    elif args.dataset == "Samsung/samsum":
        train_dataset, _ = prepare_datasets(
                args,
                "Samsung/samsum",
                tokenizer,
                lambda x: f"Dialogue:\n{x['dialogue']}\n\nSummary:",   
                lambda x: " " + x["summary"] + tokenizer.eos_token, 
                train_field="train",
                accelerator=accelerator,
            )
    elif args.dataset == "knkarthick/dialogsum":
        train_dataset, _ = prepare_datasets(
                args,
                "knkarthick/dialogsum",
                tokenizer,
                lambda x: f"Dialogue:\n{x['dialogue']}\n\nSummary:",   
                lambda x: " " + x["summary"] + tokenizer.eos_token, 
                train_field="train",
                accelerator=accelerator,
            )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


    train_dataset.save_to_disk(_path + "_train")
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    return train_dataloader
