from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

# Quantize and tokenize model

def load_model(model_name):
  """
  Configure quantization method, tokenize the model and convert pad tokens to eos tokens.
  Apply quantization to chosen model.
  return the tokenizer and quantized model.
  """

  quant_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_quant_type='nf4'
  )

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config)
  return tokenizer, model