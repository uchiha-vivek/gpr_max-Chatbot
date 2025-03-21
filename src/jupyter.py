#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip show torch


# In[2]:


get_ipython().system('pip install datasets')


# In[3]:


get_ipython().system('pip install transformers datasets peft accelerate bitsandbytes torch')


# In[5]:


from datasets import load_dataset

# Load dataset from Hugging Face
dataset = load_dataset("IraGia/gprMax_Train")

# Show first sample
print(dataset["train"][0])


# In[6]:


def format_data(example):
    return {
        "input_text": f"Instruction: {example['instruction']}\nInput: {example['input']}",
        "output_text": example["output"]
    }

# Apply transformation
dataset = dataset.map(format_data)

# Show formatted data
print(dataset["train"][0])


# In[7]:


from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose an open-source LLM
model_name = "mistralai/Mistral-7B"  # You can change this to DeepSeek or Llama

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[8]:


from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Configure LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# Define training parameters
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=1000,  # Adjust based on dataset size
    learning_rate=2e-4,
    output_dir="./gprmax_model",
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# Start Training
trainer.train()

# Save model
model.save_pretrained("./gprmax_finetuned")
tokenizer.save_pretrained("./gprmax_finetuned")


# In[ ]:


# Load fine-tuned model
model_name = "./gprmax_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate response
prompt = "write this input file\nThe receiver should be placed at 4.5696, 58.6801, 36.3701\nThe internal resistance of the voltage source is 831.6273451640435 Ohms\nThe discretisation step should be 1.5901 meters.\nI want the PML to be 9 cell thick"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
output = model.generate(inputs, max_length=100)

print(tokenizer.decode(output[0], skip_special_tokens=True))


# In[ ]:


from transformers import AutoModelForCausalLM, AutoTokenizer

# Login to Hugging Face
get_ipython().system('huggingface-cli login')

# Push to Hugging Face Model Hub
model.push_to_hub("your_username/gprMax-finetuned-model")
tokenizer.push_to_hub("your_username/gprMax-finetuned-model")

