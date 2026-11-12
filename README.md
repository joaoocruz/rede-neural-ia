# 🧠 Rede Neural Artificial — Classificação de Flores

Implementação de uma Rede Neural Artificial (RNA) desenvolvida do zero em Python, utilizando conceitos de aprendizado de máquina para classificar três espécies de flores a partir de características morfológicas. O projeto utiliza o conjunto de dados Iris e demonstra, na prática, o funcionamento de uma rede neural com uma camada escondida, incluindo propagação direta, retropropagação e ajuste de pesos por gradiente descendente.

Projeto acadêmico desenvolvido para a disciplina de Inteligência Artificial, no curso de Ciência da Computação da UNIPAC Barbacena.

---

## 1. Objetivo

Desenvolver e treinar uma Rede Neural Artificial capaz de identificar três espécies de flores do conjunto de dados Iris, utilizando quatro características morfológicas como entrada:

- Comprimento da sépala;
- Largura da sépala;
- Comprimento da pétala;
- Largura da pétala.

O principal objetivo é compreender os fundamentos matemáticos e computacionais envolvidos no funcionamento de uma rede neural, implementando suas principais operações sem utilizar bibliotecas de alto nível para construção ou treinamento de modelos.

## 2. Problema que o sistema resolve

A classificação de espécies de flores é um problema tradicional de aprendizado de máquina que consiste em identificar uma categoria a partir de características mensuráveis.

Neste projeto, a rede neural recebe informações sobre as dimensões das sépalas e pétalas de uma flor e aprende a identificar padrões que permitem distinguir três espécies:

- Iris Setosa;
- Iris Versicolor;
- Iris Virginica.

A partir dos exemplos utilizados durante o treinamento, o modelo ajusta seus parâmetros internos para reconhecer padrões e realizar previsões sobre amostras que não foram utilizadas no aprendizado.

## 3. Funcionalidades implementadas

- Carregamento do conjunto de dados Iris por meio do Scikit-Learn.
- Divisão dos dados em conjuntos de treinamento (80%) e teste (20%).
- Padronização das características utilizando média e desvio padrão.
- Conversão das classes para o formato One-Hot Encoding.
- Implementação manual de uma Rede Neural Artificial com uma camada escondida.
- Inicialização de pesos utilizando o método He Initialization.
- Implementação das funções de ativação ReLU e Softmax.
- Implementação da função de perda Cross-Entropy.
- Propagação direta (*Forward Pass*) para geração das previsões.
- Retropropagação (*Backpropagation*) para cálculo dos gradientes.
- Atualização de pesos e vieses utilizando Gradiente Descendente.
- Acompanhamento da evolução do treinamento por meio de épocas, perda e acurácia.
- Avaliação final do modelo com dados separados para teste.

## 4. Tecnologias

| Tecnologia | Uso |
|---|---|
| Python | Linguagem de programação utilizada no desenvolvimento. |
| NumPy | Operações matriciais, funções de ativação, cálculos de gradientes e atualização dos parâmetros. |
| Scikit-Learn | Carregamento do conjunto Iris e divisão dos dados em treinamento e teste. |

A biblioteca Scikit-Learn é utilizada exclusivamente para disponibilizar os dados e realizar a separação das amostras. A implementação da rede neural e de seu processo de treinamento é realizada manualmente com NumPy.

## 5. Estrutura do projeto

```text
rede-neural-ia/
├── rede-neural-aeco-inteligencia-artificial.py
└── README.md
```

O projeto possui uma estrutura simplificada, concentrando toda a implementação da rede neural em um único arquivo Python, facilitando o acompanhamento das etapas de processamento, treinamento e avaliação.

## 6. Arquitetura da rede neural

A rede neural desenvolvida possui uma camada de entrada, uma camada escondida e uma camada de saída.

| Camada | Neurônios | Função de ativação |
|---|---:|---|
| Entrada | 4 | — |
| Escondida | 8 | ReLU |
| Saída | 3 | Softmax |

**Fluxo de processamento:**

```text
4 características de entrada
           ↓
    Camada escondida
       8 neurônios
         ReLU
           ↓
     Camada de saída
       3 neurônios
        Softmax
           ↓
   Classificação da flor
```

Cada neurônio realiza operações matemáticas utilizando pesos e vieses (*bias*), permitindo que a rede aprenda relações entre as características de entrada e as classes correspondentes.

### 6.1. Funções de ativação

- **ReLU (Rectified Linear Unit):** aplicada na camada escondida, introduz não linearidade ao modelo, retornando zero para valores negativos e mantendo os valores positivos.
- **Softmax:** aplicada na camada de saída, transforma os resultados em uma distribuição de probabilidades entre as três espécies de flores.

### 6.2. Função de perda

A função Cross-Entropy é utilizada para medir a diferença entre as classes reais e as probabilidades previstas pela rede.

Durante o treinamento, o objetivo é minimizar essa perda por meio do ajuste progressivo dos pesos e vieses.

## 7. Etapas de implementação

### 7.1. Pré-processamento dos dados

O conjunto Iris é carregado utilizando a função `load_iris()`, disponibilizada pelo Scikit-Learn.

Os dados são divididos em 80% para treinamento e 20% para teste, mantendo a proporção das classes por meio da estratificação.

Em seguida, as características são padronizadas utilizando a média e o desvio padrão calculados exclusivamente sobre o conjunto de treinamento. Essa abordagem evita o vazamento de informações do conjunto de teste (*data leakage*).

As classes também são convertidas para One-Hot Encoding, permitindo que sejam utilizadas no treinamento da rede.

### 7.2. Inicialização dos parâmetros

Os pesos e vieses de cada camada são inicializados antes do treinamento.

É utilizada a técnica He Initialization para os pesos, uma estratégia adequada para redes que utilizam a função de ativação ReLU.

Uma semente aleatória fixa (`42`) também é definida para garantir a reprodutibilidade dos resultados.

### 7.3. Propagação direta (Forward Pass)

Nesta etapa, os dados de entrada percorrem as camadas da rede neural.

As operações matriciais são realizadas para calcular as ativações da camada escondida e, posteriormente, gerar as probabilidades correspondentes às três classes na camada de saída.

### 7.4. Retropropagação (Backpropagation)

Após a propagação direta, a rede calcula o erro entre as previsões e os resultados esperados.

O algoritmo de retropropagação utiliza a regra da cadeia para calcular os gradientes dos pesos e vieses de cada camada, identificando como os parâmetros devem ser ajustados para reduzir a função de perda.

### 7.5. Atualização dos parâmetros

Os parâmetros são atualizados utilizando o algoritmo de Gradiente Descendente, com uma taxa de aprendizado definida em `0.1`.

Esse processo é repetido durante 1.000 épocas, permitindo que a rede ajuste progressivamente seus parâmetros e melhore sua capacidade de classificação.

### 7.6. Avaliação do modelo

Após o treinamento, a rede realiza previsões utilizando exclusivamente o conjunto de teste.

São apresentados os seguintes indicadores:

- Perda final calculada pela Cross-Entropy;
- Quantidade de classificações corretas;
- Acurácia final do modelo.

## 8. Configurações de treinamento

| Parâmetro | Valor |
|---|---|
| Semente aleatória | 42 |
| Número de épocas | 1.000 |
| Taxa de aprendizado | 0.1 |
| Neurônios na camada escondida | 8 |
| Ativação da camada escondida | ReLU |
| Ativação da camada de saída | Softmax |
| Divisão dos dados | 80% treinamento / 20% teste |
| Função de perda | Cross-Entropy |
| Algoritmo de otimização | Gradiente Descendente |

## 9. Requisitos do ambiente

- Python 3 instalado;
- Gerenciador de pacotes Pip;
- Bibliotecas NumPy e Scikit-Learn.

## 10. Instalação e execução

### 10.1. Clonar o repositório

```bash
git clone https://github.com/joaoocruz/rede-neural-ia.git
```

### 10.2. Acessar o diretório

```bash
cd rede-neural-ia
```

### 10.3. Instalar as dependências

```bash
pip install numpy scikit-learn
```

### 10.4. Executar o projeto

```bash
python rede-neural-aeco-inteligencia-artificial.py
```

Durante a execução, o programa apresenta o progresso do treinamento, incluindo a perda e a acurácia a cada 100 épocas. Ao final, são exibidos os resultados da avaliação do modelo sobre o conjunto de teste.

## 11. Limitações da implementação

- A rede neural possui uma arquitetura fixa, com apenas uma camada escondida e oito neurônios.
- O treinamento utiliza uma taxa de aprendizado e uma quantidade de épocas previamente definidas.
- O modelo é desenvolvido especificamente para a classificação das três espécies presentes no conjunto Iris.
- Não são utilizadas bibliotecas especializadas em construção de redes neurais, como TensorFlow ou PyTorch.

## 12. Considerações finais

O desenvolvimento deste projeto permitiu explorar, de maneira prática, os fundamentos das Redes Neurais Artificiais e compreender como diferentes operações matemáticas trabalham em conjunto para permitir que um modelo aprenda padrões a partir de dados.

A implementação manual da propagação direta, retropropagação, funções de ativação e Gradiente Descendente proporcionou uma compreensão mais aprofundada do processo de treinamento, demonstrando como os parâmetros internos de uma rede neural são ajustados para realizar tarefas de classificação.

## 13. Informações acadêmicas

- **Instituição:** Centro Universitário Presidente Antônio Carlos (UNIPAC) — Barbacena.
- **Curso:** Ciência da Computação.
- **Disciplina:** Inteligência Artificial.
- **Professor:** Felipe Roncalli de Paula Carneiro.
- **Aluno:** João Pedro de Oliveira Cruz.

---

Desenvolvido para fins acadêmicos e de aprendizado.
