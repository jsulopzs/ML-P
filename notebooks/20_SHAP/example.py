import pandas as pd
import pickle
import streamlit as st

import streamlit.components.v1 as components
import shap

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)
    
with st.sidebar.form(key='house_price'):
    
    your_house = {
        'BEDROOMS': st.number_input('BEEDROOMS', value=3),
        'BATHROOMS': st.number_input('BATHROOMS', value=2),
        'GARAGE': st.number_input('GARAGE', 2),
        'FLOOR_AREA': st.number_input('FLOOR_AREA', 200),
        'BUILD_YEAR': st.number_input('BUILD_YEAR', 2000),
    }
    
    name = st.text_input('Address', placeholder='Wall Street')
    submit = st.form_submit_button('Submit')

if submit:
    
    df_house = pd.DataFrame(your_house, index=[name])
    y_pred = model.predict(df_house)[0]
    
    st.write(f'The estimated price is ${y_pred:,.2f}')
        
    shap_values = explainer.shap_values(df_house)
    st.write('Based on the model, the influence of each feature is:')
    fp = shap.force_plot(explainer.expected_value, shap_values, df_house)
    st_shap(fp, height=300)