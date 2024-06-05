import pandas as pd
import pickle
import streamlit as st

## Load trained model
import os

file_directory = os.path.dirname(os.path.abspath(__file__))
path_model = os.path.join(file_directory, 'artifacts/model.pkl')

with open(path_model, 'rb') as f:
    model = pickle.load(f)
    
# Load trained ML model to make predictions on new data
## Form data
with st.sidebar.form('price'):
    
    your_house = {
        'BEDROOMS': st.number_input('BEDROOMS', 3),
        'BATHROOMS': st.number_input('BATHROOMS', 2),
        'GARAGE': st.number_input('GARAGE', 2),
        'FLOOR_AREA': st.number_input('FLOOR_AREA', 200),
        'BUILD_YEAR': st.number_input('BUILD_YEAR', 2000)
    }
    
    submit = st.form_submit_button('Calculate')

name = 'Wall Street'
df_house = pd.DataFrame(your_house, index=[name])
    
## Calculate prediction
y_pred = model.predict(df_house)[0]

st.write(f'The estimated price is ${y_pred:,}')
st.write('Given the features of your house:')
df_house