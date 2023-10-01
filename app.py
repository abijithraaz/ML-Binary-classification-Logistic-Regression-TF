import os
import pickle
import tensorflow as tf
import gradio as gr
import pandas as pd
from sklearn.preprocessing import StandardScaler

data_heading = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value']

# Model and scaler loading
with open("./model/scaler_sklearn.pkl", "rb") as f:
    scaler = pickle.load(f)
loaded_model = tf.keras.saving.load_model('./model/house_value_model.keras')

def test_ml_model(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value):

    df_test = pd.DataFrame(data=[longitude, latitude, housing_median_age,
                            total_rooms, total_bedrooms, population,
                            households, median_income, median_house_value], columns=data_heading)
    df_test_norm = pd.DataFrame(scaler(df_test), columns=data_heading)
    
    result = loaded_model.predict(df_test_norm)
    return (f'predicted: {result}')

demo = gr.Interface(fn=test_ml_model,
                    inputs=[gr.Number(value=0.0), gr.Number(value=0.0), gr.Number(value=0.0),
                            gr.Number(value=0.0), gr.Number(value=0.0), gr.Number(value=0.0),
                            gr.Number(value=0.0), gr.Number(value=0.0), gr.Number(value=0.0),
                            gr.Number(value=0.0),], 
                    outputs="text",
                    description="A sample linear regressor solution.",
                    title='Synthetic Data Linear Regressor Solution')
    
demo.launch() 