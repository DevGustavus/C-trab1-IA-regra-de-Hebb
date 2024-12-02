package org.example.trab4;

public class v1 {
    // Função de ativação sigmoide
    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivada da sigmoide
    private static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public static void main(String[] args) {
        // Configuração da rede
        int inputNeurons = 2;   // Número de neurônios de entrada
        int hiddenNeurons = 2;  // Número de neurônios na camada oculta
        int outputNeurons = 1;  // Número de neurônios de saída
        double learningRate = 0.5;

        // Dados de entrada e saída
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] expectedOutputs = {0, 1, 1, 0}; // XOR

        // Inicialização de pesos
        double[][] inputToHiddenWeights = new double[inputNeurons][hiddenNeurons];
        double[] hiddenToOutputWeights = new double[hiddenNeurons];
        double[] hiddenBiases = new double[hiddenNeurons];
        double outputBias = Math.random();

        // Inicialização aleatória de pesos e vieses
        for (int i = 0; i < inputNeurons; i++) {
            for (int j = 0; j < hiddenNeurons; j++) {
                inputToHiddenWeights[i][j] = Math.random();
            }
        }
        for (int i = 0; i < hiddenNeurons; i++) {
            hiddenToOutputWeights[i] = Math.random();
            hiddenBiases[i] = Math.random();
        }

        // Treinamento
        for (int epoch = 0; epoch < 10000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                // Forward pass
                double[] hiddenLayerOutputs = new double[hiddenNeurons];
                for (int j = 0; j < hiddenNeurons; j++) {
                    double activation = hiddenBiases[j];
                    for (int k = 0; k < inputNeurons; k++) {
                        activation += inputs[i][k] * inputToHiddenWeights[k][j];
                    }
                    hiddenLayerOutputs[j] = sigmoid(activation);
                }

                double output = outputBias;
                for (int j = 0; j < hiddenNeurons; j++) {
                    output += hiddenLayerOutputs[j] * hiddenToOutputWeights[j];
                }
                output = sigmoid(output);

                // Erro
                double error = expectedOutputs[i] - output;

                // Retropropagação
                double outputDelta = error * sigmoidDerivative(output);
                double[] hiddenDeltas = new double[hiddenNeurons];
                for (int j = 0; j < hiddenNeurons; j++) {
                    hiddenDeltas[j] = hiddenToOutputWeights[j] * outputDelta * sigmoidDerivative(hiddenLayerOutputs[j]);
                }

                // Atualização de pesos e vieses
                for (int j = 0; j < hiddenNeurons; j++) {
                    hiddenToOutputWeights[j] += learningRate * outputDelta * hiddenLayerOutputs[j];
                }
                outputBias += learningRate * outputDelta;

                for (int j = 0; j < hiddenNeurons; j++) {
                    for (int k = 0; k < inputNeurons; k++) {
                        inputToHiddenWeights[k][j] += learningRate * hiddenDeltas[j] * inputs[i][k];
                    }
                    hiddenBiases[j] += learningRate * hiddenDeltas[j];
                }
            }
        }

        // Teste
        System.out.println("Resultados após o treinamento:");
        for (int i = 0; i < inputs.length; i++) {
            double[] hiddenLayerOutputs = new double[hiddenNeurons];
            for (int j = 0; j < hiddenNeurons; j++) {
                double activation = hiddenBiases[j];
                for (int k = 0; k < inputNeurons; k++) {
                    activation += inputs[i][k] * inputToHiddenWeights[k][j];
                }
                hiddenLayerOutputs[j] = sigmoid(activation);
            }

            double output = outputBias;
            for (int j = 0; j < hiddenNeurons; j++) {
                output += hiddenLayerOutputs[j] * hiddenToOutputWeights[j];
            }
            output = sigmoid(output);

            System.out.printf("Entrada: %d %d -> Saída prevista: %.4f%n",
                    (int) inputs[i][0], (int) inputs[i][1], output);
        }
    }
}
