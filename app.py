import streamlit as st
from transformers import pipeline
import pandas as pd

# load data
model_list = ["distilbert-base-uncased-finetuned-sst-2-english", "cardiffnlp/twitter-roberta-base-sentiment", "finiteautomata/bertweet-base-sentiment-analysis"]
model_name = st.selectbox('Which model to predict?', model_list)
text = st.text_input("text for sentiment analysis", value="This is gonna be good")

# run to get result
if st.button('Run'):
    sentiment_pipeline = pipeline(model=model_name)
    result = sentiment_pipeline(text)
    st.write(result)
