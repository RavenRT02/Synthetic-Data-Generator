from dotenv import load_dotenv
import os
from huggingface_hub import login


def login_huggingface(token=None):
    """
    Authenticate with Hugging Face. Required for Llama3.1

    If a token is provided, it is used directly (colab secrets).
    Otherwise, the function attempts to load HF_TOKEN from a local .env file.

    Args: token (str | None): Hugging Face access token.

    Raises: ValueError: If no token is provided or found.
    """

    if token is None:
        load_dotenv()
        token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError(
            "Hugging Face token not found. "
            "Provide a token or set HF_TOKEN in the .env file. "
            "or import user data from google colab."
        )
    
    login(token=token, add_to_git_credential=False)

    print("Successfully authenticated with Hugging Face")