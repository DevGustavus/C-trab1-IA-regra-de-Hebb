package org.example.rec;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import org.example.trab6.Main;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Rec {
    private int inputNodes, hiddenNodes, outputNodes;
    private double[][] weightsInputHidden, weightsHiddenOutput;
    private double[] biasHidden, biasOutput;

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    public Rec(int inputNodes, int hiddenNodes, int outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        Random rand = new Random();
        weightsInputHidden = new double[inputNodes][hiddenNodes];
        weightsHiddenOutput = new double[hiddenNodes][outputNodes];

        biasHidden = new double[hiddenNodes];
        biasOutput = new double[outputNodes];

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

    public void train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                // Feedforward
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

                // Backpropagation
                double[] outputErrors = new double[outputNodes];
                for (int j = 0; j < outputNodes; j++) {
                    outputErrors[j] = targets[i][j] - finalOutputs[j];
                }

                double[] outputGradients = new double[outputNodes];
                for (int j = 0; j < outputNodes; j++) {
                    outputGradients[j] = outputErrors[j] * sigmoidDerivative(finalOutputs[j]);
                }

                double[] hiddenErrors = new double[hiddenNodes];
                for (int j = 0; j < hiddenNodes; j++) {
                    hiddenErrors[j] = 0;
                    for (int k = 0; k < outputNodes; k++) {
                        hiddenErrors[j] += outputGradients[k] * weightsHiddenOutput[j][k];
                    }
                }

                double[] hiddenGradients = new double[hiddenNodes];
                for (int j = 0; j < hiddenNodes; j++) {
                    hiddenGradients[j] = hiddenErrors[j] * sigmoidDerivative(hiddenOutputs[j]);
                }

                // Update weights and biases
                for (int j = 0; j < hiddenNodes; j++) {
                    for (int k = 0; k < outputNodes; k++) {
                        weightsHiddenOutput[j][k] += learningRate * outputGradients[k] * hiddenOutputs[j];
                    }
                }
                for (int j = 0; j < outputNodes; j++) {
                    biasOutput[j] += learningRate * outputGradients[j];
                }

                for (int j = 0; j < inputNodes; j++) {
                    for (int k = 0; k < hiddenNodes; k++) {
                        weightsInputHidden[j][k] += learningRate * hiddenGradients[k] * inputs[i][j];
                    }
                }
                for (int j = 0; j < hiddenNodes; j++) {
                    biasHidden[j] += learningRate * hiddenGradients[j];
                }
            }

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

    public static double[][] readCSV(String filename) throws IOException, CsvValidationException {
        CSVReader reader = new CSVReader(new FileReader(filename));
        List<double[]> dataList = new ArrayList<>();
        String[] nextLine;

        // Skip header
        reader.readNext();

        while ((nextLine = reader.readNext()) != null) {
            double[] data = new double[nextLine.length - 1];  // Excluindo a coluna "DATA"

            // Percorrer as colunas de dados e fazer a conversão
            for (int i = 1; i < nextLine.length; i++) {  // Começa de 1 para ignorar a coluna "DATA"
                String value = nextLine[i].replace(',', '.'); // Substitui vírgulas por pontos

                // Se for a coluna "VOLUME", tratando B (bilhões) ou M (milhões)
                if (i == nextLine.length - 1) { // A última coluna é "VOLUME"
                    if (value.endsWith("B")) {
                        value = value.replace("B", "");
                        data[i - 1] = Double.parseDouble(value) * 1_000_000_000; // Converte para bilhões
                    } else if (value.endsWith("M")) {
                        value = value.replace("M", "");
                        data[i - 1] = Double.parseDouble(value) * 1_000_000; // Converte para milhões
                    } else {
                        data[i - 1] = Double.parseDouble(value); // Valor em unidade
                    }
                } else {
                    if (value.equals("n/d") || value.equals("null") || value.isEmpty()) {
                        data[i - 1] = 0.0; // Substitui valores não disponíveis
                    } else {
                        data[i - 1] = Double.parseDouble(value);
                    }
                }
            }
            dataList.add(data);
        }

        reader.close();

        // Converte a lista para um array de double
        double[][] dataArray = new double[dataList.size()][];
        return dataList.toArray(dataArray);
    }

    public static class NormalizationParams {
        public double[] minValues;
        public double[] maxValues;
    }

    public static Main.NormalizationParams normalizeData(double[][] data) {
        double[] minValues = new double[data[0].length];
        double[] maxValues = new double[data[0].length];

        // Inicializando com valores extremos
        for (int i = 0; i < data[0].length; i++) {
            minValues[i] = Double.MAX_VALUE;
            maxValues[i] = Double.MIN_VALUE;
            for (int j = 0; j < data.length; j++) {
                if (data[j][i] < minValues[i]) {
                    minValues[i] = data[j][i];
                }
                if (data[j][i] > maxValues[i]) {
                    maxValues[i] = data[j][i];
                }
            }
        }

        // Normalizar os dados
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = (data[i][j] - minValues[j]) / (maxValues[j] - minValues[j]);
            }
        }

        Main.NormalizationParams params = new Main.NormalizationParams();
        params.minValues = minValues;
        params.maxValues = maxValues;

        return params;
    }

    // Função para desnormalizar a previsão
    public static double desnormalize(double predictedValue, double minValue, double maxValue) {
        return predictedValue * (maxValue - minValue) + minValue;
    }

    // Função para desnormalizar e ajustar unidade (milhões ou bilhões)
    public static double desnormalizeVolume(double predictedValue, double minValue, double maxValue) {
        double realValue = predictedValue * (maxValue - minValue) + minValue;
        if (realValue >= 1_000_000_000) {  // Se o valor for em bilhões ou maior
            realValue /= 1_000_000_000;  // Converte para bilhões
            System.out.printf("Predição do VOLUME em bilhões: %.2f bilhões%n", realValue);
        } else if (realValue >= 1_000_000) {  // Se for em milhões
            realValue /= 1_000_000;  // Converte para milhões
            System.out.printf("Predição do VOLUME em milhões: %.2f milhões%n", realValue);
        } else {
            System.out.printf("Predição do VOLUME (em unidades): %.2f%n", realValue);
        }
        return realValue;  // Retorna em bilhões ou milhões
    }

    // Função para determinar a ação recomendada com base na previsão
    public static String getActionRecommendation(double predictedValue, double previousValue) {
        if (predictedValue > previousValue) {
            return "Comprar";  // A ação subirá, recomenda-se comprar
        } else {
            return "Vender";   // A ação cairá, recomenda-se vender
        }
    }

    public static void main(String[] args) throws IOException, CsvValidationException {
        // Lendo dados do arquivo CSV
        double[][] data = readCSV("C:\\Users\\gusta\\Desktop\\java-trabs-IA-intelig-comp\\trab-ia\\src\\main\\java\\org\\example\\rec\\Petrobras PETR4 - Histórico  InfoMoney.csv");
//        double[][] data = readCSV("C:\\Users\\Alvin\\Documents\\Github\\python-trab1-IA-regra-de-Hebb\\trab-ia\\src\\main\\java\\org\\example\\trab6\\VALE3.csv");

        // Normalizando os dados e armazenando os parâmetros de normalização
        Main.NormalizationParams params = normalizeData(data);

        // Dividindo os dados em entradas (inputs) e saídas (targets)
        double[][] inputs = new double[data.length - 1][6];  // Remove o último dia
        double[][] targets = new double[data.length - 1][2];  // FECHAMENTO do próximo dia

        for (int i = 0; i < data.length - 1; i++) {
            for (int j = 0; j < data[0].length; j++) {
                inputs[i][j] = data[i][j];  // Todas as entradas
            }
            targets[i][0] = data[i + 1][1];  // FECHAMENTO do próximo dia
            targets[i][1] = data[i + 1][5];  // VOLUME do próximo dia
        }

        // Criando a rede neural
        Rec nn = new Rec(6, 10, 2);  // 6 inputs (ABERTURA, FECHAMENTO, VARIAÇÃO, MÍNIMO, MÁXIMO, VOLUME)

        // Treinando a rede
        nn.train(inputs, targets, 100000, 0.7);

        // Pegando os dados do último dia
        double[] lastDayInput = new double[data[0].length];
        System.arraycopy(data[data.length - 1], 0, lastDayInput, 0, data[0].length);

        // Normalizando os dados do último dia
        for (int i = 0; i < lastDayInput.length; i++) {
            lastDayInput[i] = (lastDayInput[i] - params.minValues[i]) / (params.maxValues[i] - params.minValues[i]);
        }

        // Fazendo a previsão para o próximo FECHAMENTO e VOLUME
        double[] predictedNormalized = nn.predict(lastDayInput);

        double predictedRealValue = desnormalize(predictedNormalized[0], params.minValues[1], params.maxValues[1]);
        double predictedVolume = desnormalizeVolume(predictedNormalized[1], params.minValues[5], params.maxValues[5]);

        System.out.printf("Predição para o próximo FECHAMENTO (em BRL): %.2f%n", predictedRealValue);

        // Pegando o valor do último fechamento (último valor conhecido)
        double previousClosingPrice = desnormalize(lastDayInput[0], params.minValues[1], params.maxValues[1]);

        // Obter a recomendação de ação (comprar ou vender) com base na previsão
        String action = getActionRecommendation(predictedRealValue, previousClosingPrice);
        System.out.println("\nUltimo fechamento: " + previousClosingPrice);
        System.out.println("Recomendação: " + action);
    }
}
