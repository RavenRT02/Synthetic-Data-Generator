import json
from config import MAX_OUTPUT_TOKENS, SAFETY_MARGIN

# Find unique records

def add_batch(batch: list[dict], records: list, unique_keys: set[str]) -> None:
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

def calculate_batch_size(avg_tokens_per_record: float, max_output_tokens: int = MAX_OUTPUT_TOKENS, safety_margin: int = SAFETY_MARGIN) -> int:
  """
  calculates available tokens after negating safety_margin to account estimate inconsistencies.
  Find batch size to safely generate responses in batches efficiently.
  """

  available = max_output_tokens - safety_margin

  batch_size = int(available // avg_tokens_per_record)

  return max(1,batch_size)