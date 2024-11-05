import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

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
    
    # Seleciona as colunas de interesse (incluindo 'VOLUME', agora com 10 variáveis por dia)
    dados = df[['ABERTURA', 'FECHAMENTO', 'VARIAÇÃO', 'MÍNIMO', 'MÁXIMO', 'VOLUME']].values
    return dados, df

# Carregar e processar os dados do arquivo "ValeFinal.csv"
dados_treino, dados_combinados = carregar_dados("Financeiro/ValeFinal.csv")

# Normalização
scaler = MinMaxScaler()  # Usando MinMaxScaler
dados_normalizados = scaler.fit_transform(dados_treino)

# Definir X e y (usar janela deslizante de N dias)
n_dias_entrada = 1  # Usaremos N dias de dados para prever o próximo
X = []
y = []

for i in range(n_dias_entrada, len(dados_normalizados)):
    X.append(dados_normalizados[i-n_dias_entrada:i, :])  # Últimos N dias (agora com 10 variáveis por dia)
    y.append(dados_normalizados[i, 1])  # Prevendo o fechamento (índice 1)

X = np.array(X)
y = np.array(y)

# Dividir em treino e teste (validação cruzada para avaliar a performance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Ajuste de hiperparâmetros utilizando GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(128, 64), (256, 128), (64, 32)],  # Vários tamanhos de camadas ocultas
    'activation': ['relu', 'tanh'],  # Funções de ativação
    'solver': ['adam', 'lbfgs'],  # Solvers diferentes
    'learning_rate_init': [0.001, 0.01],  # Taxa de aprendizado
    'max_iter': [500, 1000],  # Número de iterações
}

# Usando GridSearchCV para encontrar a melhor combinação de parâmetros
modelo = MLPRegressor(random_state=42)
grid_search = GridSearchCV(modelo, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Flatten X_train para ser usado no MLPRegressor

# Melhor modelo encontrado pelo GridSearchCV
melhor_modelo = grid_search.best_estimator_

# Realizar predição
y_pred = melhor_modelo.predict(X_test.reshape(X_test.shape[0], -1))  # Flatten X_test para a predição

# Calcular o erro
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE (Erro Quadrático Médio): {rmse:.4f}")
print(f"Melhores hiperparâmetros: {grid_search.best_params_}")

# Para prever o fechamento do próximo dia (com os últimos N dias)
ultimo_dia = dados_normalizados[-n_dias_entrada:].reshape(1, n_dias_entrada * 6)  # Agora estamos usando 6 variáveis por dia (N dias * 6 variáveis)
predicao_normalizada = melhor_modelo.predict(ultimo_dia)

# Agora, vamos reverter a normalização. Como estamos prevendo o "FECHAMENTO", passamos um vetor com 6 variáveis.
predicao = scaler.inverse_transform([[0, predicao_normalizada[0], 0, 0, 0, 0]])[0][1]  # Revertendo a normalização

print(f"Previsão do fechamento para o próximo dia (amanhã): {predicao:.2f}")
