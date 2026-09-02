from llm.client import load_llm
from ui import create_ui


def main():

    llm = load_llm()
    app = create_ui(llm=llm)

    app.launch(debug=True, inbrowser=True)



if __name__ == "__main__":
    main()