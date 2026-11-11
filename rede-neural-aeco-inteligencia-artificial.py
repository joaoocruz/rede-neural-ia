"""
Atividade Aula 10 - Inteligência Artificial
Aluno: João Pedro de Oliveira Cruz 221-002302

Rede Neural Artificial com uma camada escondida, para
classificar as espécies de flores do conjunto de dados Iris.

- NumPy: todas as operações matriciais (forward, backward, atualização)
- Scikit-Learn: carregar (load_iris) e dividir (train_test_split) dados

Arquitetura: 4 entradas -> 8 neurônios (ReLU) -> 3 neurônios (Softmax)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Hiperparâmetros
# ----------------------------------------------------------------------

SEED = 42                   # semente para o resultado ser reproduzível
N_EPOCAS = 1000             # número de épocas de treinamento
TAXA_APRENDIZADO = 0.1
N_ENTRADAS = 4              # 4 características do Iris
N_ESCONDIDOS = 8            # neurônios da camada escondida
N_CLASSES = 3               # 3 espécies de flores

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# Funções de ativação
# ----------------------------------------------------------------------

# ReLU: devolve z se z > 0, senão 0.
def relu(z):
    return np.maximum(0, z)

# Derivada da ReLU: 1 onde z > 0, 0 caso contrário.
def relu_derivada(z):
    return (z > 0).astype(float)

# Softmax por linha (cada linha = uma amostra).
# Subtrai o máximo de cada linha para evitar overflow numérico.
def softmax(z):
    z_estavel = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_estavel)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ----------------------------------------------------------------------
# Função de perda
# ----------------------------------------------------------------------

# Cross-Entropy Loss média: -(1/m) * soma(y * log(y_previsto)).
def cross_entropy(y_real, y_previsto):
    m = y_real.shape[0]
    
    # O 1e-12 evita log(0).
    return -np.sum(y_real * np.log(y_previsto + 1e-12)) / m

# ----------------------------------------------------------------------
# 1. Pré-processamento dos dados
# ----------------------------------------------------------------------

iris = load_iris()
X = iris.data              # shape (150, 4)
y = iris.target            # shape (150,) com valores 0, 1, 2

# Divisão 80% treino / 20% teste 
# (estratificada para manter a proporção das classes)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Padronização (média 0 e desvio padrão 1).

""" A média e desvio são calculados somente com o treino e aplicados também
ao teste, assim nenhuma informação do teste "vaza" para o treinamento. """
media = X_treino.mean(axis=0)
desvio = X_treino.std(axis=0)

X_treino = (X_treino - media) / desvio
X_teste = (X_teste - media) / desvio

# One-Hot Encoding: classe 0 -> [1,0,0], classe 1 -> [0,1,0], classe 2 -> [0,0,1]
Y_treino = np.eye(N_CLASSES)[y_treino]
Y_teste = np.eye(N_CLASSES)[y_teste]

# ----------------------------------------------------------------------
# 2. Arquitetura: inicialização aleatória de pesos (W1, W2) e vieses (b1, b2)
# ----------------------------------------------------------------------

# Inicialização de He (indicada para ReLU): pesos ~ N(0, 2/n_entradas)
W1 = rng.standard_normal((N_ENTRADAS, N_ESCONDIDOS)) * np.sqrt(2.0 / N_ENTRADAS)
b1 = np.zeros((1, N_ESCONDIDOS))

W2 = rng.standard_normal((N_ESCONDIDOS, N_CLASSES)) * np.sqrt(2.0 / N_ESCONDIDOS)
b2 = np.zeros((1, N_CLASSES))

# ----------------------------------------------------------------------
# Forward pass
# ----------------------------------------------------------------------

def forward(X, W1, b1, W2, b2):
    # Calcula as ativações da camada escondida e da camada de saída.
    Z1 = X @ W1 + b1       # (m, 8)  entrada da camada escondida
    A1 = relu(Z1)          # (m, 8)  saída da camada escondida
    Z2 = A1 @ W2 + b2      # (m, 3)  entrada da camada de saída
    A2 = softmax(Z2)       # (m, 3)  probabilidades de cada classe
    return Z1, A1, Z2, A2

# ----------------------------------------------------------------------
# 3. Treinamento via Gradiente Descendente
# ----------------------------------------------------------------------

m = X_treino.shape[0]  # número de amostras de treino

print("Treinando a rede neural...\n")
for epoca in range(1, N_EPOCAS + 1):
    # --- Forward Pass ---
    Z1, A1, Z2, A2 = forward(X_treino, W1, b1, W2, b2)

    # --- Perda (Cross-Entropy) ---
    perda = cross_entropy(Y_treino, A2)

    # --- Backward Pass (regra da cadeia) ---
    # Softmax + Cross-Entropy juntas resultam em um gradiente simples: (A2 - Y)
    dZ2 = (A2 - Y_treino) / m              # (m, 3)
    dW2 = A1.T @ dZ2                       # (8, 3)
    db2 = np.sum(dZ2, axis=0, keepdims=True)   # (1, 3)

    dA1 = dZ2 @ W2.T                       # (m, 8) erro propagado para a escondida
    dZ1 = dA1 * relu_derivada(Z1)          # (m, 8) aplica a derivada da ReLU
    dW1 = X_treino.T @ dZ1                 # (4, 8)
    db1 = np.sum(dZ1, axis=0, keepdims=True)   # (1, 8)

    # --- Atualização dos pesos: parâmetro = parâmetro - taxa * gradiente ---
    W1 -= TAXA_APRENDIZADO * dW1
    b1 -= TAXA_APRENDIZADO * db1
    W2 -= TAXA_APRENDIZADO * dW2
    b2 -= TAXA_APRENDIZADO * db2

    # Acompanhamento do treinamento
    if epoca == 1 or epoca % 100 == 0:
        acc_treino = np.mean(np.argmax(A2, axis=1) == y_treino)
        print(f"Época {epoca:4d}/{N_EPOCAS} | Perda: {perda:.4f} | "
              f"Acurácia (treino): {acc_treino * 100:.2f}%")

# ----------------------------------------------------------------------
# 4. Avaliação do modelo no conjunto de teste
# ----------------------------------------------------------------------

_, _, _, A2_teste = forward(X_teste, W1, b1, W2, b2)
previsoes = np.argmax(A2_teste, axis=1)       # classe com maior probabilidade
acuracia = np.mean(previsoes == y_teste)
perda_teste = cross_entropy(Y_teste, A2_teste)

print("\n--- Avaliação no conjunto de teste ---")
print(f"Perda (teste): {perda_teste:.4f}")
print(f"Acertos: {np.sum(previsoes == y_teste)} de {len(y_teste)}")
print(f"Acurácia final: {acuracia * 100:.2f}%")
