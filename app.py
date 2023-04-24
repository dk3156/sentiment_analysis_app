import streamlit as st
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
import pandas as pd
import re
import torch

labels= ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

#torch updated?
#funciton for cleaning up texts
def clean_text(text):
  text = text.lower()
  text = re.sub(r"can't","cannot",text)
  text = re.sub(r"shan't","shall not",text)
  text = re.sub(r"won't","will not",text)
  text = re.sub(r"n't"," not",text) # see the space before not. 
  text = re.sub(r"i'm","i am",text)
  text = re.sub(r"what's","what is",text)
  text = re.sub(r"let's","let us",text)
  text = re.sub(r"'re"," are",text)
  text = re.sub(r"'s"," ",text)  # space because we dont know the tense , it can be is/has anything.
  text = re.sub(r"'ve"," have",text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r"\'scuse", " excuse ", text)
  text = re.sub('\W', ' ', text)  # If the comment/word does not contain any alphabets
  text = re.sub('\s+', ' ', text) # If there are more than one whitespace simultenously, then replace them by only 1 whitespace
  text = text.strip(' ') # Removing leading and trailing white spaces
  return text

def finetune(text):
    model = "dk3156/toxic_tweets_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    finetune = DistilBertForSequenceClassification.from_pretrained(model)
    tokens = tokenizer(text, return_tensors="pt")
    res = finetune(**tokens)
    scores = torch.sigmoid(res.logits)
    output = ""
    # for i in range(len(labels)):
    #     output += labels[i] + ": " + str(scores[i]) + "\n"
    print(scores[0][0].item())
    # print(text + "\n" + output)

df_test = pd.read_csv('./sample_data/test.csv')
df_test['comment_text'] = df_test['comment_text'].apply(lambda text : clean_text(text))

for comment in df_test['comment_text'][:10]:
    print(comment)
    # finetune(comment)

st.title("Sentiment Analysis")

text = st.text_input("Enter your text", value="Your dress looks like one colorful dishcloth!")

model = st.selectbox("Select the language model", ("Fine-tuned", "Binary","Non-binary"))

if st.button('Analyze'):
    if model == "Fine-tuned":
        # classifier = pipeline(model="dk3156/toxic_tweets_model", return_all_scores=True)
        # result = classifier(text)
        # cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        # for item in result:
        #     st.write(f"Sentiment: {cols[item['label']]} with score of {item['score']}")
        finetune(text)
    elif model == "Binary":
        classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)
        result = classifier(text)
        label = result[0][0]
        score = result[0][1]
        # print(f"Sentiment: {label} with score of {score}")
    else:
        classifier = pipeline(model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        result = classifier(text)
        label = result[0][0]
        score = result[0][1]
        # print(f"Sentiment: {label} with score of {score}")


