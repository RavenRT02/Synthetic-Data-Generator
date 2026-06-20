import json
import pandas as pd

from prompts import get_system_prompt
from prompts import get_user_prompt

from utils import add_batch
from utils import calculate_batch_size


# llm call

def generate_response(tokenizer, model, messages, max_new_tokens=2048):
  """
  applies chat template to the tokenizer and converts it into tensor using pytorch.
  places the tensor in gpu using .to(cuda).
  define output parameters with temperature and do_sample for randomness.
  slice content after the input till the end to get only output.
  skip special tokens like <eos> in output and return output
  """

  inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

  # Generate stochastic outputs using temperature sampling.
  outputs = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True)

  # Slicing removes the input prompt and keeps only newly generated tokens.
  return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


# estimate tokens for 1 record to calculate batch size
# Generate a small sample first to estimate average tokens per record.
# This helps dynamically decide how many records can fit into one inference call.

def estimate_tokens_per_record(tokennizer, model, base_prompt):
  """
  Generate small set of sample records to calculate the output tokens for 1 record
  """

  sample_prompt = (
      base_prompt +
      "\n\n Generate exactly 3 records."
      "\n return only valid JSON array."
      "\n No markdown"
      "\n No explanations"
  )

  messages = [{
      "role":"user", "content":sample_prompt
  }]

  response = generate_response(tokenizer, model, messages, max_new_tokens=1024)
  print(response)

  try:
    data = json.loads(response)
    print(type(data))
    print(len(data))

    if not isinstance(data, list) or len(data) == 0:
      return 100

    output_tokens = len(tokenizer.encode(response))
    return max(1, output_tokens/len(data))

  #except Exception:
    #return 100

  except Exception as e:
    print("ERROR:", e)
    print(response)
    return 100
  

# Function that uses other helper functions to generate synthetic data records

def generate_records(tokenizer, model,domain, description, count, max_output_tokens=2048):
  """
  Repeadtedly calls the llm until it creates the required number of unique synthetic data
  or until the maximum attempts is crossed.
  """

  records = []

  # Hash each dictionary as a JSON string to efficiently detect duplicates.
  unique_keys = set()

  avg_tokens = estimate_tokens_per_record(tokenizer, model, description)

  batch_limit = calculate_batch_size(avg_tokens, max_output_tokens=max_output_tokens)

  print(f"Estimated tokens/record={avg_tokens:.1f}, "
        f"batch_size={batch_limit}")

  attempts = 0
  max_attempts = count * 5

  # Keep querying the model until enough unique records are collected or the maximum retry limit is reached.
  while len(records) < count and attempts < max_attempts:

    remaining = count - len(records)

    # Avoid generating unnecessary records when nearing the target count.
    batch_size = min(remaining, batch_limit) 

    messages = [
        {"role":"system", "content": get_system_prompt(batch_size)},
        {"role":"user", "content": get_user_prompt(domain, description)}
    ]

    response = generate_response(tokenizer, model, messages, max_new_tokens=max_output_tokens)

    try:
      batch = json.loads(response)

      if not isinstance(batch, list):
        attempts += 1
        print("Response is not a list, retrying...")
        continue

      if len(batch) == 0:
          attempts += 1
          print("Empty batch, retrying...")
          continue

      add_batch(batch, records, unique_keys)
      print(f'collected {len(records)}/{count} unique records')
      attempts+=1

    except json.JSONDecodeError:
      print("Invalid JSON, retrying...")
      attempts+=1

  if len(records) < count:
    raise RuntimeError(f'could only generate {len(records)} unique records')

  synthetic_data =  records[:count]

  df = pd.DataFrame(synthetic_data)

  # Save the generated dataset for Gradio download.
  df.to_csv("synthetic_data.csv", index=False)
  csv_path = "synthetic_data.csv"

  return df.head(10), csv_path