import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Avaliação de Veículos",
    layout="wide"
)


# Função para criar o modelo e a predição
@st.cache_data
def load_data_and_model():
    cars = pd.read_csv("avaliacao_veiculo.csv", sep=";")
    encoder = OrdinalEncoder()

    for col in cars.columns.drop('evaluation'):
        cars[col] = cars[col].astype('category')

    X_encoded = encoder.fit_transform(cars.drop('evaluation',axis=1))
    y = cars['evaluation'].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    model = CategoricalNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return encoder, model, accuracy, cars

# Dados do modelo criado
encoder, model, accuracy, cars = load_data_and_model()

# Página da APP
st.title("Classificação da Qualidade do Veículo")
st.write(f"Acurácia do modelo na validação: {accuracy:.2f}")

input_features = [
        st.selectbox("Preço de compra:", cars['buying'].unique()),
        st.selectbox("Custo de manutenção:", cars['maint'].unique()),
        st.selectbox("Número de portas:", cars['doors'].unique()),
        st.selectbox("Número de assentos:", cars['seats'].unique()),
        st.selectbox("Tamanho do porta-malas:", cars['lug_boot'].unique()),
        st.selectbox("Nível de segurança:", cars['safety'].unique()),
]

# Ação no botão processar
if st.button("Processar"):
    input_df = pd.DataFrame([input_features], columns=cars.columns.drop("evaluation"))
    input_encoded = encoder.transform(input_df)
    predict_encoded = model.predict(input_encoded)
    predict = cars['evaluation'].astype('category').cat.categories[predict_encoded][0]
    st.header(f"Resultado da avaliação: {predict}")