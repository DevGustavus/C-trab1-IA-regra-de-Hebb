package org.example.trab6;

import java.util.Random;

public class Teste {
    // Número de neurônios na camada de entrada, camada oculta e camada de saída
    private int inputNodes, hiddenNodes, outputNodes;

    // Matrizes de pesos e bias
    private double[][] weightsInputHidden, weightsHiddenOutput;
    private double[] biasHidden, biasOutput;

    // Função sigmoide
    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    // Construtor da rede neural
    public Teste(int inputNodes, int hiddenNodes, int outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        // Inicializando as matrizes de pesos e bias com valores aleatórios
        Random rand = new Random();
        weightsInputHidden = new double[inputNodes][hiddenNodes];
        weightsHiddenOutput = new double[hiddenNodes][outputNodes];

        biasHidden = new double[hiddenNodes];
        biasOutput = new double[outputNodes];

        // Preenchendo com valores aleatórios entre -1 e 1
        for (int i = 0; i < inputNodes; i++) {
            for (int j = 0; j < hiddenNodes; j++) {
                weightsInputHidden[i][j] = rand.nextDouble() * 2 - 1;
            }
        }

        for (int i = 0; i < hiddenNodes; i++) {
            for (int j = 0; j < outputNodes; j++) {
                weightsHiddenOutput[i][j] = rand.nextDouble() * 2 - 1;
            }
        }

        for (int i = 0; i < hiddenNodes; i++) {
            biasHidden[i] = rand.nextDouble() * 2 - 1;
        }

        for (int i = 0; i < outputNodes; i++) {
            biasOutput[i] = rand.nextDouble() * 2 - 1;
        }
    }

    // Método para treinar a rede neural
    public void train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                // Passo 1: Feedforward

                // Entradas para a camada oculta
                double[] hiddenInputs = new double[hiddenNodes];
                for (int j = 0; j < hiddenNodes; j++) {
                    hiddenInputs[j] = 0;
                    for (int k = 0; k < inputNodes; k++) {
                        hiddenInputs[j] += inputs[i][k] * weightsInputHidden[k][j];
                    }
                    hiddenInputs[j] += biasHidden[j];
                }

                // Ativação da camada oculta
                double[] hiddenOutputs = new double[hiddenNodes];
                for (int j = 0; j < hiddenNodes; j++) {
                    hiddenOutputs[j] = sigmoid(hiddenInputs[j]);
                }

                // Entradas para a camada de saída
                double[] finalInputs = new double[outputNodes];
                for (int j = 0; j < outputNodes; j++) {
                    finalInputs[j] = 0;
                    for (int k = 0; k < hiddenNodes; k++) {
                        finalInputs[j] += hiddenOutputs[k] * weightsHiddenOutput[k][j];
                    }
                    finalInputs[j] += biasOutput[j];
                }

                // Saída final
                double[] finalOutputs = new double[outputNodes];
                for (int j = 0; j < outputNodes; j++) {
                    finalOutputs[j] = sigmoid(finalInputs[j]);
                }

                // Passo 2: Backpropagation (cálculo dos erros)

                // Erro da camada de saída
                double[] outputErrors = new double[outputNodes];
                for (int j = 0; j < outputNodes; j++) {
                    outputErrors[j] = targets[i][j] - finalOutputs[j];
                }

                // Gradiente da camada de saída
                double[] outputGradients = new double[outputNodes];
                for (int j = 0; j < outputNodes; j++) {
                    outputGradients[j] = outputErrors[j] * sigmoidDerivative(finalOutputs[j]);
                }

                // Erro da camada oculta
                double[] hiddenErrors = new double[hiddenNodes];
                for (int j = 0; j < hiddenNodes; j++) {
                    hiddenErrors[j] = 0;
                    for (int k = 0; k < outputNodes; k++) {
                        hiddenErrors[j] += outputGradients[k] * weightsHiddenOutput[j][k];
                    }
                }

                // Gradiente da camada oculta
                double[] hiddenGradients = new double[hiddenNodes];
                for (int j = 0; j < hiddenNodes; j++) {
                    hiddenGradients[j] = hiddenErrors[j] * sigmoidDerivative(hiddenOutputs[j]);
                }

                // Passo 3: Atualização dos pesos e bias

                // Atualizando pesos e bias entre camada oculta e saída
                for (int j = 0; j < hiddenNodes; j++) {
                    for (int k = 0; k < outputNodes; k++) {
                        weightsHiddenOutput[j][k] += learningRate * outputGradients[k] * hiddenOutputs[j];
                    }
                }
                for (int j = 0; j < outputNodes; j++) {
                    biasOutput[j] += learningRate * outputGradients[j];
                }

                // Atualizando pesos e bias entre camada de entrada e camada oculta
                for (int j = 0; j < inputNodes; j++) {
                    for (int k = 0; k < hiddenNodes; k++) {
                        weightsInputHidden[j][k] += learningRate * hiddenGradients[k] * inputs[i][j];
                    }
                }
                for (int j = 0; j < hiddenNodes; j++) {
                    biasHidden[j] += learningRate * hiddenGradients[j];
                }
            }

            // Exibindo o erro a cada 1000 épocas para ver a evolução do treinamento
            if (epoch % 1000 == 0) {
                double totalError = 0;
                for (int i = 0; i < inputs.length; i++) {
                    double[] hiddenInputs = new double[hiddenNodes];
                    for (int j = 0; j < hiddenNodes; j++) {
                        hiddenInputs[j] = 0;
                        for (int k = 0; k < inputNodes; k++) {
                            hiddenInputs[j] += inputs[i][k] * weightsInputHidden[k][j];
                        }
                        hiddenInputs[j] += biasHidden[j];
                    }

                    double[] hiddenOutputs = new double[hiddenNodes];
                    for (int j = 0; j < hiddenNodes; j++) {
                        hiddenOutputs[j] = sigmoid(hiddenInputs[j]);
                    }

                    double[] finalInputs = new double[outputNodes];
                    for (int j = 0; j < outputNodes; j++) {
                        finalInputs[j] = 0;
                        for (int k = 0; k < hiddenNodes; k++) {
                            finalInputs[j] += hiddenOutputs[k] * weightsHiddenOutput[k][j];
                        }
                        finalInputs[j] += biasOutput[j];
                    }

                    double[] finalOutputs = new double[outputNodes];
                    for (int j = 0; j < outputNodes; j++) {
                        finalOutputs[j] = sigmoid(finalInputs[j]);
                    }

                    for (int j = 0; j < outputNodes; j++) {
                        totalError += Math.pow(targets[i][j] - finalOutputs[j], 2);
                    }
                }
                System.out.println("Epoch: " + epoch + " Error: " + totalError);
            }
        }
    }

    // Método para obter a saída da rede
    public double[] predict(double[] input) {
        double[] hiddenInputs = new double[hiddenNodes];
        for (int i = 0; i < hiddenNodes; i++) {
            hiddenInputs[i] = 0;
            for (int j = 0; j < inputNodes; j++) {
                hiddenInputs[i] += input[j] * weightsInputHidden[j][i];
            }
            hiddenInputs[i] += biasHidden[i];
        }

        double[] hiddenOutputs = new double[hiddenNodes];
        for (int i = 0; i < hiddenNodes; i++) {
            hiddenOutputs[i] = sigmoid(hiddenInputs[i]);
        }

        double[] finalInputs = new double[outputNodes];
        for (int i = 0; i < outputNodes; i++) {
            finalInputs[i] = 0;
            for (int j = 0; j < hiddenNodes; j++) {
                finalInputs[i] += hiddenOutputs[j] * weightsHiddenOutput[j][i];
            }
            finalInputs[i] += biasOutput[i];
        }

        double[] finalOutputs = new double[outputNodes];
        for (int i = 0; i < outputNodes; i++) {
            finalOutputs[i] = sigmoid(finalInputs[i]);
        }
        return finalOutputs;
    }

    // Testando a rede neural com um exemplo simples
    public static void main(String[] args) {
        // Criando uma rede com 2 neurônios de entrada, 2 neurônios ocultos e 1 neurônio de saída
        Teste nn = new Teste(2, 2, 1);

        // Exemplo de treinamento (entradas e saídas para a porta lógica AND)
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] targets = {{0}, {0}, {0}, {1}};

        // Treinando a rede
        nn.train(inputs, targets, 10000, 0.1);

        // Testando a rede após o treinamento
        System.out.println("Predição para [0, 0]: " + nn.predict(new double[]{0, 0})[0]);
        System.out.println("Predição para [0, 1]: " + nn.predict(new double[]{0, 1})[0]);
        System.out.println("Predição para [1, 0]: " + nn.predict(new double[]{1, 0})[0]);
        System.out.println("Predição para [1, 1]: " + nn.predict(new double[]{1, 1})[0]);
    }
}
