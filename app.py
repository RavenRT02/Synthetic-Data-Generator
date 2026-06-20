from huggingface_hub import login

import gradio as gr
import time

from model import load_model
from dataset_generator import generate_records

# for Colab
# from google.colab import userdata

# for local
# import os
# hf_token = os.getenv("HF_TOKEN")


# Wrapper function invoked by Gradio UI.
# Measures total dataset generation time and prepares outputs.

LLAMA = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer, model = load_model(LLAMA)

def gradio_generate(domain, description, count):

  print("GRADIO FUNCTION CALLED")
  print(domain)
  print(description)
  print(count)

  start_time = time.time()

  file_preview, csv_path = generate_records(
      tokenizer=tokenizer, model=model,
      domain=domain, description=description, count=int(count))

  elapsed = round((time.time() - start_time) / 60, 2)  # returns seconds so div by 60, 2 for decimal places.

  status = (
      f'Dataset generated successfully.\n'
      f'Requested count : {count}\n'
      f'Generation time : {elapsed} minutes'
  )

  return status, file_preview, csv_path


# Gradio UI

with gr.Blocks(title="Synthetic Data Generator") as gradio_ui:

  gr.Markdown(
      """
      # Synthetic Dataset Generator

      Generate synthetic datasets using Llama 3.1 8B.

      Generation may take several minutes.

      Estimated times:

      25 records ≈ 3 minutes
      50 records ≈ 5 minutes
      100 records ≈ 10 minutes
      200 records ≈ 20 minutes
      """
  )

  with gr.Row():

    with gr.Column():

      domain_input = gr.Textbox(label="Domain", placeholder="Healthcare, Education, Finance, Corporate...")

      description_input = gr.Textbox(label="Dataset Description", lines=6,
                placeholder=
                """
                Generate employee records with:
                name
                age
                salary
                joining date
                department
                """
            )
      count_input = gr.Dropdown(choices=[25,50,75,100,125,150,175,200], value=25, label="Number of Records")

      generate_button = gr.Button("Generate Dataset", variant="primary")


    with gr.Column():

      status_output = gr.Textbox(label="Status", interactive=False)

      preview_output = gr.Dataframe(label="Dataset Preview")

      download_output = gr.File(label="Download CSV")


  generate_button.click(
      fn=gradio_generate,
      inputs=[domain_input, description_input, count_input],
      outputs=[status_output, preview_output, download_output]
  )

gradio_ui.launch(debug=True, inbrowser=True)