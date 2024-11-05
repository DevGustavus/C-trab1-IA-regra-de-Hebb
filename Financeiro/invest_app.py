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

# Função para carregar e processar os dados
def carregar_dados(arquivo_csv):
    df = pd.read_csv(arquivo_csv)
    
    # Substituir 'n/d' por NaN
    df.replace('n/d', np.nan, inplace=True)
    df = df.iloc[::-1].reset_index(drop=True)  # Inverter a ordem das linhas
    
    # Substituir vírgulas por pontos nas colunas numéricas, exceto 'VOLUME'
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
    return dados_normalizados, scaler

# Carregar e processar o arquivo de treino
dados_treino, scaler = carregar_dados("Financeiro/ValeFinal.csv")

# Carregar e processar o arquivo de teste (últimos 7 dias)
dados_teste, _ = carregar_dados("Financeiro/ValeFinal.csv")

# Dividir os dados em X (features) e y (target) para treino
X_train = dados_treino[:-1]  # Todas as entradas menos o último dia
y_train = dados_treino[1:, 1]  # Próximo valor de fechamento

# Construindo o MLPRegressor
modelo = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)

# Treinando o modelo
modelo.fit(X_train, y_train)

# Preparando os dados de entrada para prever o próximo fechamento
X_teste = dados_teste[-1].reshape(1, -1)  # Usando o último dia disponível no teste

# Realizando a predição
predicao_normalizada = modelo.predict(X_teste)
predicao = scaler.inverse_transform([[0, predicao_normalizada[0], 0, 0, 0, 0]])[0][1]  # Revertendo a normalização

print(f"Previsão do fechamento para o próximo dia: {predicao:.2f}")
