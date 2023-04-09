import streamlit as st
from transformers import pipeline

name_list = ["distilbert-base-uncased-finetuned-sst-2-english", "j-hartmann/emotion-english-distilroberta-base"]
model_name = st.selectbox('Which model would you like to use?', name_list)
text = st.text_input("Input text to be analyzed", value="Oh wow, I did not know that.")

if st.button('Run'):
    classifier = pipeline(model=model_name)
    result = classifier(text)
    st.write(result)
