import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# Função para processar o valor de "VOLUME"
def converter_volume(volume):
    volume = volume.replace('.', '').replace(',', '.')
    if 'B' in volume:
        return float(volume.replace('B', '')) * 1e9  # Convertendo para bilhões
    elif 'M' in volume:
        return float(volume.replace('M', '')) * 1e6  # Convertendo para milhões
    else:
        return float(volume)

# Função para carregar e processar os dados de um arquivo CSV
def carregar_dados(arquivo_csv):
    # Carregar o arquivo e inverter a ordem das linhas
    df = pd.read_csv(arquivo_csv)
    df = df.iloc[::-1].reset_index(drop=True)  # Inverter a ordem das linhas
    
    # Substituir 'n/d' por NaN
    df.replace('n/d', np.nan, inplace=True)
    
    # Substituir vírgulas por pontos nas colunas numéricas
    df[['ABERTURA', 'FECHAMENTO', 'VARIAÇÃO', 'MÍNIMO', 'MÁXIMO']] = \
        df[['ABERTURA', 'FECHAMENTO', 'VARIAÇÃO', 'MÍNIMO', 'MÁXIMO']].replace(',', '.', regex=True).astype(float)
    
    # Converter o volume usando a função personalizada
    df['VOLUME'] = df['VOLUME'].apply(converter_volume)
    
    # Tratamento de valores ausentes: preencher ou remover linhas com NaN
    df.fillna(method='ffill', inplace=True)  # Preenche NaN com o último valor válido
    
    # Convertendo a coluna de data para o formato datetime
    df['DATA'] = pd.to_datetime(df['DATA'], dayfirst=True)
    
    # Seleciona as colunas de interesse e normaliza
    dados = df[['ABERTURA', 'FECHAMENTO', 'VARIAÇÃO', 'MÍNIMO', 'MÁXIMO', 'VOLUME']].values
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dados)
    
    return dados_normalizados, scaler, df

# Carregar e processar os dados do arquivo "ValeFinal.csv"
dados_treino, scaler, dados_combinados = carregar_dados("Financeiro/ValeFinal.csv")

# Dividir os dados em X (features) e y (target) para treino
X = dados_treino[:-1]  # Todas as entradas menos o último dia
y = dados_treino[1:, 1]  # O fechamento do próximo dia (target)

# Criar e treinar o modelo
modelo = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
modelo.fit(X, y)

# Usar o último dia para prever o fechamento do próximo dia
X_ultimo_dia = dados_treino[-1].reshape(1, -1)  # Usando os dados do último dia
predicao_normalizada = modelo.predict(X_ultimo_dia)
predicao = scaler.inverse_transform([[0, predicao_normalizada[0], 0, 0, 0, 0]])[0][1]  # Revertendo a normalização

# Exibir a previsão
print(f"Previsão do fechamento para o próximo dia (amanhã): {predicao:.2f}")

# Opcional: Exibir os dados carregados
print(f"Últimos dados carregados:\n{dados_combinados.tail()}")
