import pandas as pd
import pickle
import streamlit as st
import shap
import streamlit.components.v1 as components
import json

## Load trained model
import os

dir = os.path.dirname(os.path.abspath(__file__))
path_model = os.path.join(dir, 'artifacts/model.pkl')

with open(path_model, 'rb') as f:
    model = pickle.load(f)

path_explainer = os.path.join(dir, 'artifacts/explainer.pkl')
with open(path_explainer, 'rb') as f:
    explainer = pickle.load(f)
    
path_options = os.path.join(dir, 'src/options.json')
with open(path_options, 'r') as f:
    options = json.load(f)
    
# Load trained ML model to make predictions on new data
## Form data
## DRY: DON'T REPEAT YOURSELF

with st.sidebar.form('price'):
    
    your_house = {}
    for k,v in options.items():
        your_house[k] = st.number_input(k, value=v)
    
    submit = st.form_submit_button('Calculate')

name = 'Wall Street'
df_house = pd.DataFrame(your_house, index=[name])
    
## Calculate prediction
y_pred = model.predict(df_house)[0]

st.write(f'The estimated price is ${y_pred:,}')
st.write('Given the features of your house:')

shap_values = explainer.shap_values(df_house)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
fp = shap.force_plot(explainer.expected_value, shap_values, df_house)
st_shap(fp)