import streamlit as st
from summarizer import summarize
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from get_model import download_model

import os
import time


st.set_page_config(page_title="520 project Demo", initial_sidebar_state="auto", layout="wide")

model_dir_name = 'fine_tuned_model'
model_dir=f"{os.getcwd()}/{model_dir_name}"
google_folder_id="'1KGfBg_uCMg5AnzNcAojAhbqiO13gw6u6'"

@st.cache(allow_output_mutation=True)
def get_model():
    download_model(google_folder_id,model_dir_name)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return tokenizer, model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = get_model()
model.to(device)


def app():
    st.markdown(
        """
        ## 520 project - A News Summarizer
        """
    )
    st.sidebar.subheader("Parameters")
    max_length = st.sidebar.slider("max_length", min_value=50, max_value=150, value=100, step=1)
    top_k = st.sidebar.slider("top_k", min_value=10, max_value=50, value=30, step=1)
    top_p = st.sidebar.slider("top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)


    content = st.text_area("Enter the news", max_chars=2048)
    if st.button("Generate the summary"):
        start_message = st.empty()
        start_message.write("Generating")
        start_time = time.time()
        summary = summarize(model=model, tokenizer=tokenizer, article=content, max_length=max_length, top_k=top_k, top_p=top_p)
        end_time = time.time()
        start_message.write("Generation completed{}s".format(end_time - start_time))
        st.text_input("Summary",summary)
    else:
        st.stop()

if __name__ == '__main__':
    app()