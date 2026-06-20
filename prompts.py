def get_system_prompt(batch_size):

  system_prompt = f"""
  You are a synthetic data generator.
  Your task is to generate synthetic data based on the user's requirement.
  Do not perform any other task other than generating synthetic data.
  Generate EXACTLY {batch_size} records.

  Requirements:
  - Return ONLY valid JSON.
  - Return a JSON array.
  - Array length must be exactly {batch_size}.
  - No markdown.
  - No explanations.

  example : generate employee data

  [
    {{
      "name" : "kevin",
      "age" : 25,
      "salary" : 25000
    }},

    {{
      "name" : "wilfred",
      "age" : 27,
      "salary" : 55000
    }}

  ]
  """

  return system_prompt

def get_user_prompt(domain, description):
  user_prompt = f"""
  generate data in {domain} following : {description}.
  """

  return user_prompt