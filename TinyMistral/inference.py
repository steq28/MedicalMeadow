from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
# Carica modello LoRA e tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    "fine_tuned_model", # change with folder where u have the files
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")

# Dummy Example
question = ds['input'][0]

# Genera risposta 
inputs = tokenizer(question, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
risposta = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nInput:", question)
print("\nRisposta del modello:", risposta)
