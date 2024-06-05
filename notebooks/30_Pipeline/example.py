import streamlit as st
import json
import pickle
import pandas as pd
import os

dir = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(dir, 'src/options_categorical.json')
with open(path, 'r') as f:
    options_categorical = json.load(f)

path = os.path.join(dir, 'src/options_numerical.json')
with open(path, 'r') as f:
    options_numerical = json.load(f)

path = os.path.join(dir, 'artifacts/pipeline.pkl')
with open(path, 'rb') as f:
    model = pickle.load(f)


data_input = {}
with st.sidebar.form(key='form'):
    
    name = st.text_input('Name')

    for key, value in options_categorical.items():
        data_input[key] = st.selectbox(key, value)

    for key, value in options_numerical.items():
        data_input[key] = st.number_input(key, value=value)

    button = st.form_submit_button('Submit')
        
    
if button:
    
    st.write('Based on the information you provided:')
    df_input = pd.DataFrame(data_input, index=[name])
    
    df_input

    with st.spinner('Calculating...'):

        probs = model.predict_proba(df_input)
        prob = probs[0][1] * 100
        
        message_base = 'Your approval probability for a credit card is '
        prob_str = f'{prob:.2f}%'
        
        if prob > 50:
            message = message_base + f':green[{prob_str}]'
            st.write(message)
            st.toast('Congratulations! You are approved for a credit card!', icon='🎉')
            st.balloons()
        else:
            message = message_base + f':red[{prob_str}]'
            st.write(message)
            st.toast('Sorry! You are not approved for a credit card.', icon='😢')