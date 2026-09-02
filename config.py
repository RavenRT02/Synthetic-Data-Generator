# Model

LLM_PROVIDER = "local"
LLM_MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
BASE_URL = None 

# API calls ( 2 example sets - tested models and url as of 02/09/2026 )

# LLM_PROVIDER = "api"
# LLM_MODEL = "gpt-4.1-mini"
# BASE_URL = None                      # Replace with url for providers other than openai       

# LLM_PROVIDER = "api"
# LLM_MODEL = "gemini-3.6-flash"
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# Token constants

MAX_OUTPUT_TOKENS = 2048
SAFETY_MARGIN = 200


# Temperature value

TEMPERATURE = 0.7