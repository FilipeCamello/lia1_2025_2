import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

st.set_page_config(page_title="Probabilidade de Falhas em Equipamentos", layout="wide")

st.title("Probabilidade de Falhas em Equipamentos")

with st.sidebar:
    st.header("Principal")
    type = st.radio("Selecione o tipo de cálculo:", options=["Exata", "Menos que", "Mais que"])
    occ = st.number_input("Ocorrência Atual", min_value=1, max_value=99, value=2)
    process = st.button("Processar")

if process:
    lamb = occ
    start = lamb - 2
    end = lamb + 2
    x_vals = np.arange(start, end+1)

    if type == "Exata":
        probs = poisson.pmf(x_vals, lamb)
        title = "Probabilidades de Ocorrência - Exata"
    elif type == "Menos que":
        probs = poisson.cdf(x_vals, lamb)
        title = "Probabilidades de Ocorrência - Igual ou Menor que:"
    else:
        probs = poisson.cdf(x_vals, lamb)
        title = "Probabilidades de Ocorrência - Maior que:"

    z_vals = np.round(probs,2)
    labels = [f"{i} prob.: {p}" for i,p in zip(x_vals, z_vals)]
    
    pic, ax = plt.subplots()
    ax.bar(x_vals, probs, tick_label=labels, color=plt.cm.gray(np.linspace(0.4,0.8, len(x_vals))))
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(pic)
    
#https://appdataanalysistravel.streamlit.app/