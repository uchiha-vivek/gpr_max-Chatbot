"""Inspiration from https://docs.unsloth.ai/get-started/fine-tuning-guide"""

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None 
load_in_4bit = True  


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # any light weight open-source LLM
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
  
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 8, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,  
    bias = "none",     
    use_gradient_checkpointing = "unsloth",  
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)


# Memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()
# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
  gprmax_prompt.format(
    "create the input file",  # Prompt instruction
    "The receiver should be positioned at 5.2341, 60.7892, 38.4523. The internal resistance of the voltage source is 920.5127849236743 Ohms. The discretisation step should be 1.7432 meters. I want the PML to be 11 cells thick. The dimensions of the model are 18.2045 x 80.4321 x 42.6789 meters. The central frequency will be equal to 0.145 GHz. Introduce random geological layers. The duration of the signal will be 2.4983720183456721e-05 seconds. The polarization of the pulse should be 'y'. The pulse should be ricker. The transmitter (Tx) is a Magnetic Dipole.",  # Input
    "",  # Output - leave this blank for generation!
)

], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


# Prompting

# gprmax_prompt
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[gprmax_prompt.format(
    "Build this model",  # Instruction
    "Generate stratified layers. The dimensions of the model are 3.1257 x 45.7892 x 70.3628 meters. The excitation pulse should be gaussianmodulated. Tx is an Electric Dipole. I want the PML to be 85 cells thick. The duration of the signal will be 9.872134281321675e-06 seconds. The central frequency will be equal to 4025.112 MHz. The source should be placed at the coordinates 0.5123, 22.1345, 52.6789 meters. The internal resistance of the voltage source is 715.3248796134521 Ohms. The receiver should be positioned at (0.5123, 22.1345, 52.6789) meters. The polarization of the pulse should be 'y'.",  # Input
    "",  # Output - leave this blank for generation!
)

], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)



model.save_pretrained("gpr_fine_tune_model")  
tokenizer.save_pretrained("gpr_fine_tune_model")




if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "gpr_fine_tune_model", 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  

# gprmax_prompt = You MUST copy from above!

inputs = tokenizer(
[
   gprmax_prompt.format(
    "Name a well-known ancient pyramid in Egypt?",  # Instruction
    "",  # Input
    "",  # Output - leave this blank for generation!
)

], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)