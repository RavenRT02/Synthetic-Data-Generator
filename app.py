# Intended method to run the app 
# For Llama3.1 -- run colab_app.ipynb in google colab T4 compute
# For models through api calls (openai, anthropic, etc..) run this file locally after altering config.py accordingly

from llm.client import load_llm
from ui import create_ui


def main():

    llm = load_llm()
    app = create_ui(llm=llm)

    app.launch(debug=True, inbrowser=True)



if __name__ == "__main__":
    main()