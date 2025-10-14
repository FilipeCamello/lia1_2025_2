import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Configuração da página
st.set_page_config(page_title="Previsão de Inflação", layout="wide")
st.title("📈 Previsão de Inflação Mensal - Brasil")

# Sidebar
st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("📂 Faça upload do CSV", type=["csv"])

if uploaded_file is not None:
    # Ler o CSV
    df = pd.read_csv(uploaded_file)
    
    # Mostrar dados originais
    st.write("### 📊 Dados Originais")
    st.dataframe(df, use_container_width=True)
    
    # Processamento SIMPLES dos dados
    dados_lista = []
    
    for _, row in df.iterrows():
        ano = row['Year']
        # Verificar TODAS as colunas que podem ser meses
        for coluna in df.columns:
            if coluna != 'Year' and coluna != 'Total' and coluna != 'Year ':
                valor = row[coluna]
                if pd.notna(valor) and valor != '':
                    dados_lista.append({
                        'Ano': ano,
                        'Mes': coluna.strip(),  # Remove espaços extras
                        'Valor': float(valor)
                    })
    
    # Criar DataFrame limpo
    df_clean = pd.DataFrame(dados_lista)
    
    if len(df_clean) == 0:
        st.error("❌ Não foi possível extrair dados do arquivo.")
        st.stop()
    
    # Ordem dos meses
    ordem_meses = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Converter mês para categórico para ordenar
    df_clean['Mes'] = pd.Categorical(df_clean['Mes'], categories=ordem_meses, ordered=True)
    df_clean = df_clean.sort_values(['Ano', 'Mes']).reset_index(drop=True)
    
    # Criar labels para o gráfico - CORREÇÃO DO ERRO
    abreviacoes = {'January': 'Jan', 'February': 'Fev', 'March': 'Mar', 'April': 'Abr',
                  'May': 'Mai', 'June': 'Jun', 'July': 'Jul', 'August': 'Ago',
                  'September': 'Set', 'October': 'Out', 'November': 'Nov', 'December': 'Dez'}
    
    # CORREÇÃO: Converter para string antes de concatenar
    df_clean['Mes_Str'] = df_clean['Mes'].astype(str)
    df_clean['AnoMes'] = df_clean['Mes_Str'].map(abreviacoes) + '/' + df_clean['Ano'].astype(str)
    
    # Mostrar dados processados
    st.write("### 🔄 Dados Processados")
    st.write(f"Total de registros: {len(df_clean)}")
    st.dataframe(df_clean[['AnoMes', 'Valor']].head(12))
    
    # Configurações de previsão
    st.sidebar.write("---")
    st.sidebar.subheader("Parâmetros de Previsão")
    
    total_meses = len(df_clean)
    
    meses_treino = st.sidebar.number_input(
        "Meses para treino:",
        min_value=1,
        max_value=total_meses,
        value=min(36, total_meses),
        step=1
    )
    
    meses_previsao = st.sidebar.number_input(
        "Meses para previsão:",
        min_value=1,
        max_value=24,
        value=6,
        step=1
    )
    
    if st.sidebar.button("🔮 Gerar Previsão"):
        # Dados para treino
        valores_treino = df_clean['Valor'].tail(meses_treino).values
        
        try:
            # Modelo ARIMA
            model = ARIMA(valores_treino, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Previsão
            forecast = model_fit.forecast(steps=meses_previsao)
            
            # Gerar datas futuras
            ultimo_ano = df_clean['Ano'].iloc[-1]
            ultimo_mes = df_clean['Mes'].iloc[-1]
            ultimo_idx = ordem_meses.index(ultimo_mes)
            
            datas_futuras = []
            ano_atual = ultimo_ano
            idx_mes = ultimo_idx
            
            for i in range(meses_previsao):
                idx_mes += 1
                if idx_mes >= 12:
                    idx_mes = 0
                    ano_atual += 1
                datas_futuras.append(f"{abreviacoes[ordem_meses[idx_mes]]}/{ano_atual}")
            
            # DataFrames para mostrar
            df_treino = pd.DataFrame({
                'Período': df_clean['AnoMes'].tail(meses_treino).values,
                'Valor Real': valores_treino
            })
            
            df_prev = pd.DataFrame({
                'Período': datas_futuras,
                'Previsão': forecast
            })
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📋 Dados de Treino")
                st.dataframe(df_treino, use_container_width=True)
            
            with col2:
                st.write("### 🔮 Previsão")
                st.dataframe(df_prev, use_container_width=True)
            
            # Gráfico
            st.write("### 📈 Histórico + Previsão")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Histórico em azul
            ax.plot(df_treino['Período'], df_treino['Valor Real'], 
                   label='Histórico', color='blue', marker='o', linewidth=2)
            
            # Previsão em vermelho
            ax.plot(df_prev['Período'], df_prev['Previsão'], 
                   label='Previsão', color='red', marker='s', linestyle='--', linewidth=2)
            
            ax.set_xlabel("Período")
            ax.set_ylabel("Inflação (%)")
            ax.set_title("Previsão de Inflação")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erro: {e}")

else:
    st.info("📤 Faça upload do arquivo CSV para começar")