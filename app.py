import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Dataset
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import nltk
import gensim
from sentence_transformers import CrossEncoder
#cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)

from PIL import Image
# Loading Image using PIL
im = Image.open('dashboard.png')
# Adding Image to web app
st.set_page_config(page_title="Report Recommendation App", page_icon = im, layout="wide")

#st.title('Report Recommendation App')

st.markdown("<h1 style='text-align: center; color: black;'>Report Recommendation App</h1>", unsafe_allow_html=True)

st.markdown('''<p style='text-align: center; color: blue;'><b>This application helps the user identify the right report they are looking for.</b></p>''',
            unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return tokenizer, cross_model, summarizer


df = pd.read_csv('input_data.csv',encoding='latin1')
df.dropna(inplace=True)

df_corpus = df['KPI Descriptions']
df_corpus = pd.DataFrame(df_corpus)
ds = Dataset.from_pandas(df_corpus)

tokenizer, cross_model, summarizer = get_model()

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
    #reply_message = 'The model is fetching relevant reports for the given context -' + user_input
    #st.write(reply_message)
    st.write("Fetching the Results")

    model_inputs = [[user_input,item['KPI Descriptions']] for item in results]
    scores = cross_score(model_inputs)
    #Sort the scores in decreasing order
    ranked_results = [{'Dashboard Name': inp['Dashboard Name'], 'Score': score} for inp, score in zip(results, scores)]
    ranked_results = sorted(ranked_results, key=lambda x: x['Score'], reverse=True)
    final_result = pd.DataFrame(ranked_results)

    import datetime
    today = datetime.datetime.today()
    day_of_week = today.strftime("%A")

    general_chatbot_convs = [
        {'prompt':'What is today?',
        'response':f'Today is {day_of_week}'},
        {'prompt':'What is this app about?',
        'response':f'This app provides the recommendation on the reports based on user natural language query. It also provides impotant KPIs of these dashboard in a summarized format.'},
        {'prompt':'How many reports are there in the data?',
        'response':f'Total number of reports in the list is {df.shape[0]}'},
        {'prompt':'Who is the creator of this app?',
        'response':f'The creator of this app is Lakshminarayanan'},
        {'prompt':'Why is this app created?',
        'response':f'This app is the demonstration of my dissertation thesis'}  
    ]
    general_convs_df = pd.DataFrame(general_chatbot_convs)

    org_results_gen = general_convs_df[['prompt', 'response']].to_dict(orient='records')
    model_inputs_gen = [[user_input,item['prompt']] for item in org_results_gen]
    scores_gen = cross_score(model_inputs_gen)
    #Sort the scores in decreasing order
    ranked_results_gen = [{'prompt': inp['prompt'], 'Score': score} for inp, score in zip(org_results_gen, scores_gen)]
    ranked_results_gen = sorted(ranked_results_gen, key=lambda x: x['Score'], reverse=True)
    ranked_results_gen = pd.DataFrame(ranked_results_gen)

    #final_results = pd.DataFrame()
    #final_results['cross_encoder'] = [item['Dashboard Name'] for item in ranked_results[0:3]]

    df_result = df[df['Dashboard Name'].isin([item['Dashboard Name'] for item in ranked_results[0:2]])]
    df_result.rename(columns = {'KPI Description': 'About the Dashboard', 'KPI Values':'Key Insights'}, inplace = True)
    df_result.drop(columns='KPIs', inplace = True)
    df_result.reset_index(inplace=True)
    df_result.drop(columns='index', inplace = True)

    if final_result.head(1).Score[0] > 0.001:
        dash_name = final_result.head(1)['Dashboard Name'][0]
        kpis = df.loc[df['Dashboard Name'] == dash_name]['KPI Values']
        prompt_res = df.loc[df['Dashboard Name'] == dash_name]['KPI Descriptions']
        prompt_res = prompt_res.tolist()[0]
        summary = summarizer(prompt_res, max_length=100, min_length=30, do_sample=False)
        prompt_res = summary[0]['summary_text']
        st.markdown("<h2 style='text-align: left; color: black;'>Results</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: left; color: black;'><b>Relevant Report<b></h4>", unsafe_allow_html=True)
        st.write(f"The Name of the Relevant Report is **{dash_name}**")
        st.write(f"**Important KPI results of the dashboard** {kpis}")
        st.write(f"**Description** {prompt_res}")

    elif ranked_results_gen.head(1).Score[0] > 0.1:
        prompt_res = ranked_results_gen.head(1)['prompt'][0]
        prompt_res = general_convs_df.loc[general_convs_df['prompt'] == prompt_res]['response']
        prompt_res = prompt_res.tolist()[0]
        st.write(prompt_res)
    else:
        prompt_res = "Cannot find any relevance in the list, kindly enter a different Prompt"
        st.markdown(prompt_res, unsafe_allow_html=True)
