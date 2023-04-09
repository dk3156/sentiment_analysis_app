import streamlit as st
from transformers import pipeline
import pandas as pd

name_list = ["distilbert-base-uncased-finetuned-sst-2-english", "cardiffnlp/twitter-roberta-base-sentiment", "cardiffnlp/twitter-xlm-roberta-base-sentiment"]
model_name = st.selectbox('Which model would you like to use?', name_list)
text = st.text_input("Please input text to analyze", value="This is gonna be good")

if st.button('Run'):
    sentiment_pipeline = pipeline(model=model_name)
    result = sentiment_pipeline(text)
    st.write(result)
