import streamlit as st
from transformers import pipeline

name_dict = {"Binary" : "distilbert-base-uncased-finetuned-sst-2-english", "Non-binary" : "j-hartmann/emotion-english-distilroberta-base"}
model_name = st.selectbox('Which model would you like to use?', name_dict.keys())
text = st.text_input("Enter your text!", value="Oh wow, I did not know that.")

if st.button('Analyze'):
    classifier = pipeline(model=name_dict[model_name], return_all_scores=True)
    result = classifier(text)
    for i in range len(result[0]):
        st.write(result[0][i])

