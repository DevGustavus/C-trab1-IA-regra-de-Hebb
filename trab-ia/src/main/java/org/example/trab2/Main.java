package org.example.trab2;

import javax.swing.*;
import java.awt.*;
import java.util.Random;
import java.util.Scanner;

// ================================= PERCEPTRON =================================
public class Main {

    float[][] entrada = new float[2][4];
    int opc = 0;
    float limiar = 0;
    float[] w = new float[4]; // Um peso para cada entrada.
    int[] target = new int[2]; // Duas possíveis saídas.
    float b;
    float alfa = 0.01f; // Taxa de aprendizagem 0 < X <= 1.
    float yLiq = 0;
    int contCiclo = 0;
    float y;
    float yTeste;
    int lin, col;
    int condErro = 1;
    float teste;

    // Construtor
    public Main() {
        // Inicializando os alvos
        target[0] = -1; // Reconhecimento de A.
        target[1] = 1;  // Reconhecimento de B.

        // Inicializando os pesos
        w[0] = 0.5f;
        w[1] = -0.1f;
        w[2] = 0.4f;
        w[3] = -0.27f;

        // Inicializando o bias
        b = 0.3256f;

        // Inicializando as entradas
        entrada[0][0] = -1;
        entrada[0][1] = -1;
        entrada[0][2] = 1;
        entrada[0][3] = 1;

        entrada[1][0] = 1;
        entrada[1][1] = -1;
        entrada[1][2] = 1;
        entrada[1][3] = -1;
    }

    public void executarPerceptron() {
        Scanner scanner = new Scanner(System.in);

        while (opc != 3) {
            System.out.println("\n\n ***** Programa Perceptron *****");
            System.out.println("Digite 1 para treinar a rede");
            System.out.println("Digite 2 para operar");
            System.out.println("Digite 3 para Sair");
            System.out.print("-> ");
            opc = scanner.nextInt();

            if (opc == 1) {
                while (condErro == 1) {
                    condErro = 0;
                    lin = 0;

                    // Aqui lin deve ser incrementado somente após o fim de cada ciclo de treinamento.
                    while (lin < 2) {
                        yLiq = 0;
                        col = 0;

                        while (col < 4) {
                            yLiq += entrada[lin][col] * w[col];
                            col++; // Incrementa a coluna após o cálculo
                        }

                        // Adiciona o bias após o cálculo de todas as entradas
                        yLiq += b;

                        // Função de ativação
                        if (yLiq >= limiar) {
                            y = 1;
                        } else {
                            y = -1;
                        }

                        // Exibe o valor calculado e o alvo esperado
                        System.out.printf("\n y: %.2f - target: %.2f", y, (float) target[lin]);

                        // O segredo do perceptron está aqui: verifica se há erro
                        if (y != target[lin]) {
                            condErro = 1;
                            col = 0;

                            // Algoritmo para correção dos pesos e bias
                            while (col < 4) {
                                w[col] += alfa * (target[lin] - y) * entrada[lin][col];
                                col++;
                            }
                            // Atualiza o bias com a mesma lógica
                            b += alfa * (target[lin] - y);
                        }

                        // Incrementa lin apenas aqui para passar para o próximo conjunto de dados
                        lin++;
                    }
                    // Exibe o número do ciclo
                    System.out.printf("\n Ciclo: %d \n", contCiclo);
                    contCiclo++;
                }
                System.out.println("\n Rede treinada!");

                // Exibe os pesos
                col = 0;
                while (col < 4) {
                    System.out.printf("\n Peso [%d]: %.2f", col, w[col]);
                    col++;
                }
                // Exibe o valor do bias
                System.out.printf("\n Bias: %.2f", b);
            }

            if (opc == 2) {
                System.out.printf("\n\t ----> Testando a rede treinada");
                System.out.printf("\n\t Teste com as entradas do treinamento");

                // Mostrando as entradas
                lin = 0;
                while (lin < 2) {
                    col = 0;
                    while (col < 4) {
                        System.out.printf("\n Entrada [%d] [%d]: %.2f", lin, col, entrada[lin][col]);
                        col++;
                    }
                    lin++;
                    System.out.printf("\n -------------------");
                }

                // Teste da rede
                for (lin = 0; lin < 2; lin++) {
                    teste = 0;

                    for (col = 0; col < 4; col++) { // Somatório dos pesos
                        teste += entrada[lin][col] * w[col];
                    }
                    teste += b;

                    if (teste >= limiar) {
                        yTeste = 1;
                    } else {
                        yTeste = -1;
                    }

                    System.out.printf("\n Saida da rede [%d]: %.2f", lin, yTeste);
                }
            }
        }

        scanner.close();
        System.out.println("Programa encerrado.");
    }

    public static void main(String[] args) {
        Main perceptron = new Main();
        perceptron.executarPerceptron();
    }

}
