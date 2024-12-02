package org.example.trab5;

import java.util.Random;

public class Main {
    public static void main(String[] args) {
        // Configurações da rede neural
        int entradas = 1;
        int neur = 200;
        double alfa = 0.01;
        double erroTolerado = 0.01;
        double xmin = -1.0;
        double xmax = 1.0;
        int nPontos = 50;

        // Gerando entradas igualmente espaçadas
        double[] xOrig = new double[nPontos];
        double step = (xmax - xmin) / (nPontos - 1);
        for (int i = 0; i < nPontos; i++) {
            xOrig[i] = xmin + i * step;
        }

        double[][] x = new double[nPontos][1];
        for (int i = 0; i < nPontos; i++) {
            x[i][0] = xOrig[i];
        }

        // Gerando o target
        double[][] t = new double[1][nPontos];
        for (int i = 0; i < nPontos; i++) {
            t[0][i] = Math.sin(x[i][0]) * Math.sin(2 * x[i][0]);
        }

        // Inicializando pesos sinápticos
        Random random = new Random();
        double[][] v = new double[entradas][neur];
        double[] v0 = new double[neur];
        double[][] w = new double[neur][1];
        double[] w0 = new double[1];

        for (int i = 0; i < entradas; i++) {
            for (int j = 0; j < neur; j++) {
                v[i][j] = random.nextDouble() * 2 - 1; // Aleatório entre -1 e 1
            }
        }

        for (int j = 0; j < neur; j++) {
            v0[j] = random.nextDouble() * 2 - 1;
        }

        for (int i = 0; i < neur; i++) {
            w[i][0] = random.nextDouble() * 0.4 - 0.2; // Aleatório entre -0.2 e 0.2
        }

        w0[0] = random.nextDouble() * 0.4 - 0.2;

        // Variáveis auxiliares
        double[][] zin = new double[1][neur];
        double[][] z = new double[1][neur];
        double[][] yin;
        double[][] y = new double[1][1];
        double erroTotal = 1.0;
        int ciclo = 0;

        // Treinamento
        while (erroTolerado < erroTotal) {
            erroTotal = 0;
            for (int padrao = 0; padrao < nPontos; padrao++) {
                // Forward pass
                for (int j = 0; j < neur; j++) {
                    zin[0][j] = v0[j];
                    for (int k = 0; k < entradas; k++) {
                        zin[0][j] += x[padrao][k] * v[k][j];
                    }
                    z[0][j] = Math.tanh(zin[0][j]);
                }

                double yinTemp = w0[0];
                for (int j = 0; j < neur; j++) {
                    yinTemp += z[0][j] * w[j][0];
                }
                y[0][0] = Math.tanh(yinTemp);

                // Cálculo do erro
                double erro = t[0][padrao] - y[0][0];
                erroTotal += 0.5 * erro * erro;

                // Backpropagation
                double deltinhaK = erro * (1 - y[0][0] * y[0][0]); // Derivada de tanh
                double[] deltinhaJ = new double[neur];
                for (int j = 0; j < neur; j++) {
                    deltinhaJ[j] = deltinhaK * w[j][0] * (1 - z[0][j] * z[0][j]);
                }

                // Atualização dos pesos
                for (int j = 0; j < neur; j++) {
                    w[j][0] += alfa * deltinhaK * z[0][j];
                }
                w0[0] += alfa * deltinhaK;

                for (int j = 0; j < neur; j++) {
                    for (int k = 0; k < entradas; k++) {
                        v[k][j] += alfa * deltinhaJ[j] * x[padrao][k];
                    }
                    v0[j] += alfa * deltinhaJ[j];
                }
            }

            ciclo++;
            System.out.println("Ciclo: " + ciclo + ", Erro Total: " + erroTotal);
        }

        // Teste
        double[] yTeste = new double[nPontos];
        for (int i = 0; i < nPontos; i++) {
            for (int j = 0; j < neur; j++) {
                zin[0][j] = v0[j];
                for (int k = 0; k < entradas; k++) {
                    zin[0][j] += x[i][k] * v[k][j];
                }
                z[0][j] = Math.tanh(zin[0][j]);
            }

            double yinTemp = w0[0];
            for (int j = 0; j < neur; j++) {
                yinTemp += z[0][j] * w[j][0];
            }
            yTeste[i] = Math.tanh(yinTemp);
        }

        // Impressão dos resultados (gráfico seria implementado externamente)
        System.out.println("x, Target, Output");
        for (int i = 0; i < nPontos; i++) {
            System.out.println(xOrig[i] + ", " + t[0][i] + ", " + yTeste[i]);
        }
    }
}
