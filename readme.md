# Exploration Repository

This repository contains code that can run Random Network Distillation with Mario and ALE/Montezuma (Atari game).

To run the experiment, it is preferable to have a custom environment set up, such as `virtualenv` or `conda`.

```bash
cd exploration
pip install -r requirements.txt
pip install -e .
```

For Mario, use the following command:

```bash
python mario_ppo.py
```
Supervised Finetuning using TRL library

```
python examples/scripts/vsft_llava.py \
    --model_name_or_path="llava-hf/llava-1.5-7b-hf" \
    --report_to="wandb" \
    --learning_rate=1.4e-5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --output_dir="data/vsft-llava-1.5-7b-hf" \
    --logging_steps=5 \
    --num_train_epochs=1 \
    --push_to_hub \
    --gradient_checkpointing \
    --remove_unused_columns=False \
    --torch_dtype=float16 \
    --fp16=True
    
# peft:
python examples/scripts/vsft_llava.py \    
    --model_name_or_path="llava-hf/llava-1.5-7b-hf" \
    --report_to="wandb" \
    --learning_rate=1.4e-5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --output_dir="data/vsft-llava-1.5-7b-hf" \
    --logging_steps=5 \
    --num_train_epochs=1 \
    --push_to_hub \
    --gradient_checkpointing \
    --remove_unused_columns=False \
    --torch_dtype=float16 \
    --fp16=True \ 
    --use_peft=True \
    --lora_r=64 \
    --lora_alpha=16 \
    --lora_target_modules=all-linear"

# evaluation:
 
To evaluate, first install the lmms-eval framework: pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
then run:
accelerate launch --num_processes=8 -m lmms_eval \
        --model llava_hf \
        --model_args pretrained=llava-hf/llava-1.5-7b-hf \
        --tasks mmbench \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_sample    
```
