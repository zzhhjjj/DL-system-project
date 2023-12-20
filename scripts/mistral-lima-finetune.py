import transformers
from datetime import datetime
import os
from transformers import Trainer, TrainingArguments
from peft import PeftModel  
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

from datasets import load_dataset

# File paths
training_file_path = "data/processed/lima/training_set.jsonl"
validation_file_path = "data/processed/lima/validation_set.jsonl"

# Load the training dataset
train_dataset = load_dataset('json', data_files=training_file_path, split='train')

# Load the validation dataset
eval_dataset = load_dataset('json', data_files=validation_file_path, split='train')


base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=1024,
    padding_side="left",
    add_special_tokens=True
)
tokenizer.pad_token = tokenizer.eos_token

def format_and_tokenize(data_point):
    """
    Formats and tokenizes the input and output text.
    
    Args:
    input_text (str): The user input text.
    output_text (str): The assistant's output text.
    tokenizer: The tokenizer to use for tokenization.
    11
    Returns:
    dict: A dictionary with tokenized 'input_ids' and 'attention_mask'.
    """
    # Format the text
    formatted_text = f"<s>[INST] {data_point['input']} [/INST] {data_point['output']}</s>"

    # Tokenize the formatted text
    tokenized_output = tokenizer(formatted_text, truncation=True, padding="max_length")

    return tokenized_output

tokenized_train_dataset = train_dataset.map(format_and_tokenize)
tokenized_val_dataset = eval_dataset.map(format_and_tokenize)

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
# print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

num_train_examples = len(tokenized_train_dataset) 
gradient_accumulation_steps = 4
num_epochs = 50
per_device_train_batch_size=16
num_training_steps = (num_train_examples // per_device_train_batch_size) * num_epochs

### Restart from the checkpoint
project = "lima-finetune-v3"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name


tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=0,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_steps=num_training_steps,
        weight_decay=0.1,
        lr_scheduler_type="linear",  # Linear decay of learning rate
        adam_beta1=0.9,
        adam_beta2=0.95,
        learning_rate=1e-5, # Want about 10x smaller than the Mistral learning rate
        logging_steps=100,
        bf16=True,   #True for bf16
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=100,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=100,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# checkpoint_path = "mistral-lima-finetune-v2/checkpoint-2500"
# trainer.train(resume_from_checkpoint=checkpoint_path)
trainer.train()















































