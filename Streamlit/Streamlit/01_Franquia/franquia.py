import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Previsão de Custo para Franquia")

# Carregar os dados
data = pd.read_csv("franquia_custos_iniciais.csv", sep=";")

# X -> dataframe; y -> série do pandas
X = data[['custo_franquia_anual']]
y = data['investimento_inicial']

# Criar o modelo
model = LinearRegression().fit(X,y)

col1, col2 = st.columns(2)

with col1:
    st.header("Dados")
    st.table(data.head())

    col1, col2 = st.columns(2)

# Tratar coluna 1 - dados
with col1:
    st.header("Dados")
    st.table(data.head())
    
# Tratar coluna 2 - gráfico de dispersão
with col2:
    st.header("Gráfico de Dispersão")
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue")
    ax.plot(X, model.predict(X), color="red")
    st.pyplot(fig)
    
st.header("Valor Anual da Franquia")
new_value = st.number_input("Insira valor em R$", min_value=1.0, max_value=99999.0, value=1500.0, step=1.00)
process = st.button("Processar")

if process:
   data_new_value = pd.DataFrame([[new_value]], columns=["custo_franquia_anual"])
   prevision = model.predict(data_new_value)
   st.header(f"Previsão de Custo: R$ {prevision[0]:.2f}") 

# Para rodar a APP
# python - m streamlit run .\01. Franquia.py