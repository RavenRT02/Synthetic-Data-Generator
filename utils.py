import json

# Find unique records

def add_batch(batch, records, unique_keys):
  """
  Find duplicate dictionaries in batch list.
  Append only unique dictionaries / records from the batch.
  """

  for record in batch:
    key = json.dumps(record, sort_keys=True)

    if key not in unique_keys:
      unique_keys.add(key)
      records.append(record)


# calculate batch size

def calculate_batch_size(avg_tokens_per_record, max_output_tokens=2048, safety_margin=200):
  """
  calculates available tokens after negating safety_margin to account estimate inconsistencies.
  Find batch size to safely generate responses in batches efficiently.
  """

  available = max_output_tokens - safety_margin

  batch_size = int(available // avg_tokens_per_record)

  return max(1,batch_size)