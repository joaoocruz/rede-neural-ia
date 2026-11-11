# Artificial Neural Network — Iris Classification

Implementation of an Artificial Neural Network (ANN) with one hidden layer, developed in Python to classify flower species using the Iris dataset.

This project was developed as an academic assignment for the Artificial Intelligence course, with the goal of implementing and training a neural network from scratch, using NumPy for matrix operations and Scikit-Learn exclusively for data loading and preprocessing.

## Objective

Develop a neural network capable of identifying three flower species based on four features:

- Sepal length;
- Sepal width;
- Petal length;
- Petal width.

## Neural Network Architecture

The network consists of one hidden layer with the following configuration:

| Layer | Neurons | Activation Function |
|---|---:|---|
| Input | 4 | — |
| Hidden | 8 | ReLU |
| Output | 3 | Softmax |

The output layer represents the three classes in the Iris dataset.

## Technologies

- **Python:** programming language used for development.
- **NumPy:** matrix operations, activation functions, gradient calculations, and parameter updates.
- **Scikit-Learn:** loading the Iris dataset and splitting it into training and testing sets.

## Implementation Steps

### 1. Data Preprocessing

- Load the Iris dataset using `load_iris()`;
- Split the dataset into 80% training and 20% testing data;
- Standardize input features using a mean of 0 and a standard deviation of 1;
- Convert class labels using One-Hot Encoding.

### 2. Neural Network Implementation

- Random initialization of weights and biases;
- He initialization for weights;
- Implementation of ReLU and Softmax activation functions;
- Implementation of the Cross-Entropy loss function.

### 3. Training

The network is trained using Gradient Descent, following two main stages:

- **Forward Pass:** propagates input data through the network to generate predictions.
- **Backward Pass:** applies Backpropagation to calculate gradients and update weights and biases using the chain rule.

### 4. Model Evaluation

After training, the network makes predictions using the test dataset. The following metrics are displayed:

- Final loss (*Cross-Entropy Loss*);
- Number of correct predictions;
- Final model accuracy.

## Training Configuration

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Training epochs | 1,000 |
| Learning rate | 0.1 |
| Hidden layer neurons | 8 |
| Hidden layer activation | ReLU |
| Output layer activation | Softmax |
| Training/testing split | 80% / 20% |

## Getting Started

### Prerequisites

- Python installed;
- Pip package manager.

### Install Dependencies

Install the required libraries:

```bash
pip install numpy scikit-learn
```

### Run the Project

Execute the following command in your terminal:

```bash
python rede-neural-aeco-inteligencia-artificial.py
```

During execution, the program displays training progress and, at the end, the model evaluation results.

## Project Structure

```text
.
├── rede-neural-aeco-inteligencia-artificial.py
└── README.md
```

## Academic Information

**Institution:** Centro Universitário Presidente Antônio Carlos (UNIPAC) — Barbacena, Brazil

**Course:** Computer Science

**Subject:** Artificial Intelligence

**Professor:** Felipe Roncalli de Paula Carneiro

**Student:** João Pedro de Oliveira Cruz