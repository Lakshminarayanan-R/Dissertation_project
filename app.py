import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Dataset
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import nltk
import gensim
from sentence_transformers import CrossEncoder
#cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)

from PIL import Image
# Loading Image using PIL
im = Image.open('dashboard.png')
# Adding Image to web app
st.set_page_config(page_title="Surge Price Prediction App", page_icon = im)

st.title('Report Recommendation App')

st.text('A compant has lots of dashboards and reports and it is important to get insights from the right report. This application helps the user identify the right report they are looking for.')

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)
    return tokenizer, cross_model


df = pd.read_csv('input_data.csv',encoding='latin1')
df.dropna(inplace=True)

df_corpus = df['KPI Descriptions']
df_corpus = pd.DataFrame(df_corpus)
ds = Dataset.from_pandas(df_corpus)

tokenizer, cross_model = get_model()

def tokenize_function(train_dataset):
    return tokenizer(train_dataset['KPI Descriptions'], padding='max_length', truncation=True) 
tokenized_dataset = ds.map(tokenize_function, batched=True)
results = df[['Dashboard Name', 'KPI Descriptions']].to_dict(orient='records')

def cross_score(model_inputs):
    scores = cross_model.predict(model_inputs)
    return scores


user_input = st.text_area("Provide a context of the Report that you wish to look")
button = st.button("Fetch Report List")


if user_input and button:
    st.write('The model is fetching relevant reports for the given context -', user_input)
    st.write("It might take a few minutes to load")
    model_inputs = [[user_input,item['KPI Descriptions']] for item in results]
    scores = cross_score(model_inputs)
    #Sort the scores in decreasing order
    ranked_results = [{'Dashboard Name': inp['Dashboard Name'], 'Score': score} for inp, score in zip(results, scores)]
    ranked_results = sorted(ranked_results, key=lambda x: x['Score'], reverse=True)

    final_results = pd.DataFrame()
    final_results['cross_encoder'] = [item['Dashboard Name'] for item in ranked_results[0:3]]

    df_result = df[df['Dashboard Name'].isin([item['Dashboard Name'] for item in ranked_results[0:3]])]
    df_result.rename(columns = {'KPI Description': 'About the Dashboard', 'KPI Values':'Key Insights'}, inplace = True)
    df_result.drop(columns='KPIs', inplace = True)
    df_result

    st.table(final_results)
