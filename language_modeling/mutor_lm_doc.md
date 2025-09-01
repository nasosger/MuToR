# 🧠 Language Modeling with MuToR

This folder contains the code for finetuning language models with **MuToR** on downstream generative tasks,  such as mathematical reasoning. Below we provide a complete guide for setting up the environment, training models, and performing inference.

---

## 1. ⚙️ Environment Setup

We recommend using a conda environment with Python 3.10. 
```bash

conda create -n mutor_lm python=3.10
conda activate mutor
```
Follow these commands to install the necessary dependencies. We include the version for transformers and accelerate as well.

```bash 
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/nasosger/MuToR.git
cd MuToR/language_modeling
pip install -r requirements.txt
```
Our experiments with the 2B model were run using a single **A100** GPU, while for the 8B model we employed **5xA100** GPUs and **FSDP**.
<br>
Also make sure that you **log in to wandb** before launching.



## 2. 🚀 Finetuning with MuToR
Before launching a finetuning experiment, specify the arguments in the provided bash scripts: **bash_0.sh, bash_1.sh, launch_finetuning.sh**. These control stuff such as the model type, dataset, batch size, learning rate, and hyperparameters for MuToR. 
<br> 🛠️ **MuToR's Hyperparameters** : We recommend setting $d_{max}=4, a=0.3$, since it performed better in most cases. 
<br> ❗ **Attention** : If your hardware allows train_batch_size_per_device > 1, you need to modify the code accordingly, to handle padding. The provided code does not provide this functionality.

To launch a finetuning experiment, use the main training launcher script:

``` bash
cd language_modeling
bash scripts/launch_finetuning.sh
```

---
Another thing to notice: In our experiments with a single GPU, we hardcoded the total number of gradient updates (args.max_train_steps), and the number of warmup steps (args.num_warmup_steps). These are used when initializing the learning scheduler. To reproduce our results, set the following values:

| Dataset   | `max_train_steps` |  `num_warmup_steps` |
|-----------|-------------------|---------------------|
| GSM8K     | 3735              | 374                 |
| 1M-GSM    | 75485             | 7549                |
| 1M-MATH   | 83525             | 8353                |

--- 
For Llama 8B, setup FSDP configuration with HF accelerate:
```bash
accelerate config
```
and then proceed to launch the script.

## 3. Evaluation/Inference with MuToR
For evaluation/inference, we provide two scripts: evaluate_math500.py, evaluate_gsm8k.py. Adjust some critical hyperparameters such as the model's architecture, the path to the checkpoints, etc.
Then run the script with the following command:
``` python
CUDA_VISIBLE_DEVICES=0 python src/eval/evaluate_gsm8k.py
```

---
As far as the benchmarks are concerned:
- GSM8K is available via HF datasets. <br>
- Make sure you download MATH500 test split from [OpenAI](https://github.com/openai/prm800k/tree/main/prm800k/math_splits).

## Acknowledgement

Our training scripts were adapted from [Bitune](https://github.com/dkopi/bitune). 
For the implementation of custom models, we adapted the code from the official **HF transformers** implementations.