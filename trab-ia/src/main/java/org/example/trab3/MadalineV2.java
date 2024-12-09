package org.example.trab3;

import java.io.*;
import java.util.Random;


public class MadalineV2 {
    public static void main(String[] args) throws IOException {
        // Configuração de caminho
//        String caminho = "/home/alvin/Documentos/Github/python-trabs-IA-intelig-comp/trab-ia/src/main/java/org/example/trab3/";

        String caminho = "C:\\Users\\Alvin\\Documents\\Github\\python-trab1-IA-regra-de-Hebb\\trab-ia\\src\\main\\java\\org\\example\\trab3\\";

        // Carregamento das entradas
        double[][] entAux = loadMatrix(caminho + "Ent.txt");
        int padroes = entAux.length;
        int entradas = entAux[0].length;

        // Carregamento das saídas
        double[][] targAux = loadMatrixCSV(caminho + "Targ.csv", ";");
        int numSaidas = targAux.length;

        // Inicialização das variáveis
        double limiar = 0.0;
        double alfa = 0.1;
        double erroTolerado = 0.1;

        // Pesos e Biases
        double[][] v = new double[entradas][numSaidas];
        Random rd = new Random();
        for (int i = 0; i < entradas; i++) {
            for (int j = 0; j < numSaidas; j++) {
                v[i][j] = rd.nextDouble() * 0.02 - 0.01; // Inicializando com valores entre -0.1 e 0.1
            }
        }

        double[] v0 = new double[numSaidas];
        for (int j = 0; j < numSaidas; j++) {
            v0[j] = rd.nextDouble() * 0.2 - 0.1; // Inicializando com valores entre -0.1 e 0.1
        }

        // Saídas calculadas pela rede
        double[] yin = new double[numSaidas];
        double[] y = new double[numSaidas];

        double erro = 1.0;
        int ciclo = 0;

        // Treinamento da rede neural
        while (erro > erroTolerado) {
            ciclo++;
            erro = 0.0;
            for (int i = 0; i < padroes; i++) {
                double[] padraoLetra = entAux[i];  // Atribui o padrão de entrada

                // Cálculo da soma das entradas e pesos
                for (int m = 0; m < numSaidas; m++) {
                    double soma = 0.0;
                    for (int n = 0; n < entradas; n++) {
                        soma += padraoLetra[n] * v[n][m];
                    }
                    yin[m] = soma + v0[m];
                }

                // Função de ativação sigmoid
                for (int j = 0; j < numSaidas; j++) {
                    y[j] = sigmoid(yin[j]);
                }

                // Cálculo do erro
                for (int j = 0; j < numSaidas; j++) {
                    erro += 0.5 * Math.pow(targAux[j][i] - y[j], 2);
                }

                // Atualização dos pesos e biases
                double[][] vanterior = new double[entradas][numSaidas];
                for (int m = 0; m < entradas; m++) {
                    for (int n = 0; n < numSaidas; n++) {
                        vanterior[m][n] = v[m][n];
                    }
                }

                for (int m = 0; m < entradas; m++) {
                    for (int n = 0; n < numSaidas; n++) {
                        v[m][n] = vanterior[m][n] + alfa * (targAux[n][i] - y[n]) * padraoLetra[m];
                    }
                }

                double[] v0anterior = v0.clone();
                for (int j = 0; j < numSaidas; j++) {
                    v0[j] = v0anterior[j] + alfa * (targAux[j][i] - y[j]);
                }
            }

            erro /= padroes;  // Normaliza o erro dividindo pela quantidade de padrões
            System.out.println("Ciclo: " + ciclo);
            System.out.println("Erro: " + erro);
        }
        // Rede treinada com sucesso - Teste de uma entrada
        double[] entTeste = entAux[new Random().nextInt(7)]; // Uma letra é desginada aleatoriamente
        for (int j = 0; j < numSaidas; j++) {
            double soma = 0.0;
            for (int i = 0; i < entradas; i++) {
                soma += entTeste[i] * v[i][j];
            }
            yin[j] = soma + v0[j];
        }

        // Exibe a saída da rede treinada
        for (int j = 0; j < numSaidas; j++) {
            if (yin[j] >= limiar) {
                y[j] = 1.0;
            } else {
                y[j] = -1.0;
            }
        }

        // Exibe os resultados de `yin` e `y`
        System.out.println("yin: ");
        for (double val : yin) {
            System.out.print(val + " ");
        }
        System.out.println();

        System.out.println("y: ");
        for (double val : y) {
            System.out.print(val + " ");
        }
        System.out.println();
    }

    // Função de ativação sigmoid
    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Função para carregar uma matriz de um arquivo de texto
    public static double[][] loadMatrix(String fileName) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        String line;
        String[] tokens;
        int rowCount = 0;

        // Contagem das linhas
        while ((line = br.readLine()) != null) {
            rowCount++;
        }

        br.close();

        // Lê novamente o arquivo para carregar os dados
        br = new BufferedReader(new FileReader(fileName));
        double[][] matrix = new double[rowCount][];
        int i = 0;

        while ((line = br.readLine()) != null) {
            tokens = line.split(" ");
            matrix[i] = new double[tokens.length];
            for (int j = 0; j < tokens.length; j++) {
                matrix[i][j] = Double.parseDouble(tokens[j]);
            }
            i++;
        }
        br.close();

        return matrix;
    }

    // Função para carregar uma matriz de um arquivo CSV
    public static double[][] loadMatrixCSV(String fileName, String delimiter) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        String line;
        String[] tokens;
        int rowCount = 0;

        // Contagem das linhas
        while ((line = br.readLine()) != null) {
            rowCount++;
        }

        br.close();

        // Lê novamente o arquivo para carregar os dados
        br = new BufferedReader(new FileReader(fileName));
        double[][] matrix = new double[rowCount][];
        int i = 0;

        while ((line = br.readLine()) != null) {
            tokens = line.split(delimiter);
            matrix[i] = new double[tokens.length];
            for (int j = 0; j < tokens.length; j++) {
                matrix[i][j] = Double.parseDouble(tokens[j]);
            }
            i++;
        }
        br.close();

        return matrix;
    }
}
