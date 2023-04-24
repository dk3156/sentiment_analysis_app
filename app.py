import streamlit as st
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
import pandas as pd
import re
import torch

labels= ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# table = {
#     "Why the edits made under my usern..." : 
# }

def finetune(text):
    model = "dk3156/toxic_tweets_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    finetune = DistilBertForSequenceClassification.from_pretrained(model)
    tokens = tokenizer(text, return_tensors="pt")
    res = finetune(**tokens)
    scores = torch.sigmoid(res.logits)
    output = ""
    print(text[:30], score[0][0].item())

    # for i in range(len(labels)):
    #     output += labels[i] + ": " + str(round(scores[0][i].item(), 1)) + " "
    # st.write(text[:20] + "... \n" + output)

df_test = pd.read_csv('./sample_data/test.csv')

for text in df_test['comment_text'][:10]:
    finetune(text)

st.title("Sentiment Analysis")

text = st.text_input("Enter your text", value="Your dress looks like one colorful dishcloth!")

model = st.selectbox("Select the language model", ("Fine-tuned", "Binary","Non-binary"))

if st.button('Analyze'):
    if model == "Fine-tuned":
        finetune(text)
    elif model == "Binary":
        classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)
        result = classifier(text)
        label = result[0][0]
        score = result[0][1]
        st.write(f"Sentiment: {label} with score of {score}")
    else:
        classifier = pipeline(model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        result = classifier(text)
        label = result[0][0]
        score = result[0][1]
        st.write(f"Sentiment: {label} with score of {score}")


