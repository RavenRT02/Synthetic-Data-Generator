from openai import OpenAI
from dotenv import load_dotenv
import os
from auth import login_huggingface
from config import LLM_MODEL, LLM_PROVIDER, BASE_URL
from model import load_model
from dataset_generator import generate_response



class LocalLLM:
    """
    Wrapper around a local hugging face model
    """

    def __init__(self, model_name: str):

        login_huggingface()
        self.tokenizer, self.model = load_model(model_name=model_name)


    def generate(self, messages: list[dict]) -> str:

        return generate_response(tokenizer=self.tokenizer, model=self.model, messages=messages)




class APIClient:
    """
    Client for OpenAI-compatible chat completions APIs
    """

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)


    def generate(self, messages: list[dict]) -> str:

        response = self.client.chat.completions.create(model=self.model_name, messages=messages)

        return response.choices[0].message.content.strip()




def load_llm():
    """
    Load the configured LLM backend based on LLM_PROVIDER config value
    """

    if LLM_PROVIDER == "local":
        return LocalLLM(model_name=LLM_MODEL)

    if LLM_PROVIDER == "api":

        load_dotenv()
        api_key = os.get("LLM_API_KEY")

        if not api_key:
            raise ValueError("API key not found. Set LLM_API_KEY in .env file.")

        return APIClient(model_name=LLM_MODEL, api_key=api_key, base_url=BASE_URL)

    raise ValueError(f'Unsupported LLM Provider {LLM_PROVIDER}')