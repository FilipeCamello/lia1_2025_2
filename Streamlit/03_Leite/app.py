import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from datetime import date
from io import StringIO

st.set_page_config(page_title="Sistema de Análise e Previsão de Séries Temporais", layout="wide")

st.title("Estimativa de Produção de Leite")

with st.sidebar:
    uploaded_file = st.file_uploader("Escolha o arquivo:", type=['csv'])
    if uploaded_file is not None:
        string_io = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = pd.read_csv(string_io, header=None)
        start_date = date(2011, 1, 1)
        period = st.date_input("Período Inicial da Série", start_date)
        forecast_period = st.number_input("Informe a quantidade de meses para previsão", 
                                           min_value=1, max_value=48, value=12)
        process_button = st.button("Processar")

if uploaded_file is not None and process_button:
    try:
        ts_data = pd.Series(data.iloc[:,0].values, index=pd.date_range(
            start=period, periods=len(data), freq='M'))
        decompose = seasonal_decompose(ts_data, model='additive')
        pic_decompose = decompose.plot()
        pic_decompose.set_size_inches(10,8)

        model = SARIMAX(ts_data, order=(2,0,0,), seasonal_order=(0,1,1,12))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_period)

        pic_forecast, ax = plt.subplots(figsize=(10,5))
        ax = ts_data.plot(ax=ax)
        forecast.plot(ax=ax, style='r--')

        col1, col2, col3 = st.columns([3,3,2])
        with col1:
            st.write("Decomposição")
            st.pyplot(pic_decompose)
        with col2:
            st.write("Previsão")
            st.pyplot(pic_forecast)
        with col3:
            st.write("Dados da Previsão")
            st.dataframe(forecast)               
    
    except Exception as ex:
        st.error(f"Erro ao processar os dados!: {ex}")