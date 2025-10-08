import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Sistema de Recomendação Dietética",
    page_icon="🍎",
    layout="wide"
)

# Função para criar o modelo
@st.cache_data
def load_data_and_model():
    # Carregar dados
    df = pd.read_csv("Dieta\diet_recommendations_reduced.csv")
    
    # Calcular BMI
    df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)
    
    # Adicionar categoria do BMI
    def classify_bmi(bmi):
        if bmi < 18.5: return 'Abaixo do Peso'
        elif 18.5 <= bmi < 25: return 'Normal'
        elif 25 <= bmi < 30: return 'Sobrepeso'
        else: return 'Obesidade'
    df['BMI_Category'] = df['BMI'].apply(classify_bmi)
    
    # Preencher valores vazios com 'None'
    df['Disease_Type'] = df['Disease_Type'].fillna('None')
    df['Dietary_Restrictions'] = df['Dietary_Restrictions'].fillna('None')
    df['Allergies'] = df['Allergies'].fillna('None')
    
    # Determinar restrição dietética automaticamente baseado na doença
    def determine_restriction(disease):
        if disease == 'Diabetes':
            return 'Low_Sugar'
        elif disease == 'Hypertension':
            return 'Low_Sodium'
        else:
            return 'None'
    
    df['Auto_Restriction'] = df['Disease_Type'].apply(determine_restriction)
    
    # Codificar variáveis categóricas
    encoder = OrdinalEncoder()
    categorical_cols = ['Gender', 'Disease_Type', 'Severity', 'Physical_Activity_Level', 
                      'Auto_Restriction', 'Allergies', 'Preferred_Cuisine', 'BMI_Category']
    
    # Garantir que todas as categorias sejam incluídas
    X_encoded = encoder.fit_transform(df[categorical_cols])
    y = df['Diet_Recommendation']
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calcular acurácia
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return encoder, model, accuracy, df, categorical_cols

# Carregar modelo e dados
try:
    encoder, model, accuracy, df, feature_cols = load_data_and_model()
    
    # Obter categorias únicas de cada coluna para os selectboxes
    gender_options = df['Gender'].unique()
    disease_options = df['Disease_Type'].unique()
    severity_options = df['Severity'].unique()
    activity_options = df['Physical_Activity_Level'].unique()
    allergy_options = df['Allergies'].unique()
    cuisine_options = df['Preferred_Cuisine'].unique()
    
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Interface da aplicação
st.title("🍎 Sistema de Recomendação Dietética")
st.write(f"**Acurácia do modelo:** {accuracy:.2f}")

st.subheader("Informe seus dados:")

col1, col2 = st.columns(2)

with col1:
    idade = st.number_input("Idade", min_value=18, max_value=100, value=35)
    peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
    altura = st.number_input("Altura (cm)", min_value=100.0, max_value=220.0, value=170.0)

with col2:
    genero = st.selectbox("Gênero", gender_options)
    atividade = st.selectbox("Nível de Atividade", activity_options)
    alergias = st.selectbox("Alergias", allergy_options)

# Nova linha para doença e gravidade
col3, col4 = st.columns(2)

with col3:
    doenca = st.selectbox("Condição de Saúde", disease_options)

with col4:
    # Gravidade só aparece se uma doença for selecionada
    if doenca != 'None':
        gravidade = st.selectbox("Gravidade da Condição", severity_options)
    else:
        gravidade = 'Mild'  # Valor padrão quando não há doença
        st.selectbox("Gravidade da Condição", severity_options, disabled=True, help="Selecione uma condição de saúde para habilitar")

# Culinária em linha própria
culinaria = st.selectbox("Culinária Preferida", cuisine_options)

# Calcular BMI
bmi = peso / ((altura / 100) ** 2)
if bmi < 18.5: 
    bmi_categoria = 'Abaixo do Peso'
elif 18.5 <= bmi < 25: 
    bmi_categoria = 'Normal'
elif 25 <= bmi < 30: 
    bmi_categoria = 'Sobrepeso'
else: 
    bmi_categoria = 'Obesidade'

st.write(f"**Seu IMC:** {bmi:.1f} ({bmi_categoria})")

# Determinar restrição dietética automaticamente
def get_auto_restriction(disease):
    if disease == 'Diabetes':
        return 'Low_Sugar'
    elif disease == 'Hypertension':
        return 'Low_Sodium'
    else:
        return 'None'

auto_restriction = get_auto_restriction(doenca)

# Mostrar restrição automática se aplicável
if auto_restriction != 'None':
    st.info(f"🔍 **Restrição dietética automática:** {auto_restriction} (baseado na condição de saúde)")

# Botão de predição
if st.button("🎯 Obter Recomendação de Dieta", type="primary"):
    # Preparar dados de entrada
    input_features = [genero, doenca, gravidade, atividade, auto_restriction, alergias, culinaria, bmi_categoria]
    input_df = pd.DataFrame([input_features], columns=feature_cols)
    
    try:
        # Fazer predição
        input_encoded = encoder.transform(input_df)
        predict = model.predict(input_encoded)[0]
        
        # Descrições das dietas
        diet_info = {
            'Balanced': {
                'nome': 'Dieta Balanceada',
                'descricao': 'Dieta equilibrada com todos os grupos alimentares, proporcionando nutrientes essenciais para manutenção da saúde geral e bem-estar.',
                'motivos': []
            },
            'Low_Carb': {
                'nome': 'Dieta Low Carb', 
                'descricao': 'Dieta com redução de carboidratos, focada no controle glicêmico e favorecendo a perda de peso saudável.',
                'motivos': []
            },
            'Low_Sodium': {
                'nome': 'Dieta com Baixo Sódio',
                'descricao': 'Dieta com restrição de sal e alimentos processados, ideal para controle da pressão arterial e saúde cardiovascular.',
                'motivos': []
            }
        }
        
        # Determinar motivos baseados nas entradas do usuário
        info = diet_info[predict]
        
        # Adicionar motivos baseados nas características do usuário
        if predict == 'Low_Carb':
            if doenca == 'Diabetes':
                info['motivos'].append('Condição de Diabetes requer controle glicêmico')
            if bmi_categoria in ['Sobrepeso', 'Obesidade']:
                info['motivos'].append('IMC indica necessidade de controle de peso')
            if auto_restriction == 'Low_Sugar':
                info['motivos'].append('Restrição automática de açúcar alinhada com dieta low carb')
            if len(info['motivos']) < 3:
                info['motivos'].append('Nível de atividade favorece metabolismo de gorduras')
                
        elif predict == 'Low_Sodium':
            if doenca == 'Hypertension':
                info['motivos'].append('Condição de Hipertensão requer controle de sódio')
            if gravidade in ['Moderate', 'Severe']:
                info['motivos'].append('Gravidade da condição exige cuidado redobrado')
            if auto_restriction == 'Low_Sodium':
                info['motivos'].append('Restrição automática de sódio para saúde cardiovascular')
            if len(info['motivos']) < 3:
                info['motivos'].append('Perfil de saúde beneficia-se da redução de sal')
                
        else:  # Balanced
            if doenca == 'None':
                info['motivos'].append('Ausência de condições de saúde específicas')
            elif doenca == 'Obesity':
                info['motivos'].append('Condição de obesidade com abordagem balanceada para perda de peso sustentável')
            if bmi_categoria == 'Normal':
                info['motivos'].append('IMC dentro da faixa saudável')
            elif bmi_categoria in ['Sobrepeso', 'Obesidade']:
                info['motivos'].append('Abordagem balanceada para controle de peso sustentável')
            if atividade in ['Moderate', 'Active']:
                info['motivos'].append('Nível de atividade compatível com dieta balanceada')
        
        # Garantir que temos exatamente 3 motivos
        while len(info['motivos']) < 3:
            info['motivos'].append('Perfil individual favorece esta abordagem dietética')
        info['motivos'] = info['motivos'][:3]
        
        # Mostrar resultados
        st.success("✅ Recomendação gerada com sucesso!")
        
        st.header(f"🍽️ {info['nome']}")
        st.write(f"**Descrição:** {info['descricao']}")
        
        st.subheader("🎯 Principais motivos para esta recomendação:")
        for i, motivo in enumerate(info['motivos'], 1):
            st.write(f"{i}. {motivo}")
        
        # Informações adicionais
        with st.expander("📊 Detalhes do seu perfil"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dados Pessoais:**")
                st.write(f"- Idade: {idade} anos")
                st.write(f"- Gênero: {genero}")
                st.write(f"- IMC: {bmi:.1f} ({bmi_categoria})")
            with col2:
                st.write("**Condições de Saúde:**")
                st.write(f"- Condição: {doenca}")
                if doenca != 'None':
                    st.write(f"- Gravidade: {gravidade}")
                st.write(f"- Atividade: {atividade}")
                if auto_restriction != 'None':
                    st.write(f"- Restrição automática: {auto_restriction}")
                st.write(f"- Alergias: {alergias}")
                st.write(f"- Culinária: {culinaria}")
                
    except Exception as e:
        st.error(f"Erro ao gerar recomendação: {e}")
        st.info("Verifique se todas as opções selecionadas estão presentes nos dados de treinamento.")

# Rodapé
st.markdown("---")
st.markdown("*Sistema de recomendação baseado em aprendizado de máquina para orientação dietética personalizada.*")