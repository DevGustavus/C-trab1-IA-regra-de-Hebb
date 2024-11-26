package org.example.trab2;

import javax.swing.*;
import java.awt.*;
import java.util.Random;

public class Main {

    private static final int SIZE = 10; // Tamanho da matriz (10x10)
    private int[][] firstLetterMatrix = new int[SIZE][SIZE];
    private int[][] secondLetterMatrix = new int[SIZE][SIZE];
    private int[][] testMatrix = new int[SIZE][SIZE];
    private double[][] weights = new double[SIZE][SIZE]; // Pesos para cada célula
    private double bias = 0; // Bias do Perceptron
    private final double LEARNING_RATE = 0.05; // Taxa de aprendizado ajustada
    private final int EPOCHS = 5000; // Número de épocas ajustado

    public static void main(String[] args) {
        SwingUtilities.invokeLater(Main::new);
    }

    public Main() {
        JFrame frame = new JFrame("Perceptron - Reconhecimento de Letras");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 800);
        frame.setLayout(new BorderLayout());

        // Painel principal
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new GridLayout(3, 1, 10, 10)); // Três painéis empilhados, com espaçamento

        // Criação das três matrizes
        JPanel firstLetterPanel = createMatrixPanel(firstLetterMatrix, "Letra 1");
        JPanel secondLetterPanel = createMatrixPanel(secondLetterMatrix, "Letra 2");
        JPanel testPanel = createMatrixPanel(testMatrix, "Teste");

        mainPanel.add(firstLetterPanel);
        mainPanel.add(secondLetterPanel);
        mainPanel.add(testPanel);

        // Botões
        JPanel buttonPanel = new JPanel();
        JButton trainButton = new JButton("Treinar");
        JButton testButton = new JButton("Testar");

        // Eventos dos botões
        trainButton.addActionListener(e -> trainPerceptron());
        testButton.addActionListener(e -> testRecognition());

        buttonPanel.add(trainButton);
        buttonPanel.add(testButton);

        frame.add(mainPanel, BorderLayout.CENTER);
        frame.add(buttonPanel, BorderLayout.SOUTH);

        frame.setVisible(true);
        frame.setResizable(false);

        initializeWeights(); // Inicializa os pesos aleatórios
    }

    private JPanel createMatrixPanel(int[][] matrix, String title) {
        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(SIZE, SIZE));
        panel.setBorder(BorderFactory.createTitledBorder(title));

        int cellSize = 40;
        Dimension buttonSize = new Dimension(cellSize, cellSize);

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                JButton button = new JButton();
                button.setBackground(Color.WHITE);
                button.setFocusable(false);
                button.setPreferredSize(buttonSize);
                final int row = i, col = j;

                button.addActionListener(e -> togglePixel(button, matrix, row, col));
                panel.add(button);
            }
        }

        panel.setPreferredSize(new Dimension(SIZE * cellSize, SIZE * cellSize));
        return panel;
    }

    private void togglePixel(JButton button, int[][] matrix, int row, int col) {
        if (matrix[row][col] == 0) {
            matrix[row][col] = 1;
            button.setBackground(Color.BLACK);
        } else {
            matrix[row][col] = 0;
            button.setBackground(Color.WHITE);
        }
    }

    private void initializeWeights() {
        Random random = new Random();
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                weights[i][j] = random.nextDouble() * 0.1 - 0.05; // Inicializa pesos com valores pequenos
            }
        }
        bias = random.nextDouble() * 0.1 - 0.05; // Inicializa o bias com valor pequeno
    }

    private void trainPerceptron() {
        // Treina com Letra 1 (d = 1) e Letra 2 (d = -1)
        trainSingleMatrix(firstLetterMatrix, 1);
        trainSingleMatrix(secondLetterMatrix, -1);

        JOptionPane.showMessageDialog(null, "Treinamento concluído!", "Perceptron", JOptionPane.INFORMATION_MESSAGE);
    }

    private void trainSingleMatrix(int[][] letterMatrix, int desiredOutput) {
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            int output = predict(letterMatrix);  // Predição
            int error = desiredOutput - output;  // Erro

            if (error != 0) {  // Se houver erro, ajusta os pesos
                for (int i = 0; i < SIZE; i++) {
                    for (int j = 0; j < SIZE; j++) {
                        weights[i][j] += LEARNING_RATE * error * letterMatrix[i][j];  // Ajuste dos pesos
                    }
                }
                bias += LEARNING_RATE * error;  // Ajuste do viés
            }
        }
    }

    private int predict(int[][] inputMatrix) {
        double sum = 0.0;
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                sum += weights[i][j] * inputMatrix[i][j];  // Soma ponderada
            }
        }
        sum += bias;  // Soma do viés

        // Função de ativação: se a soma for maior ou igual a zero, retorna 1 (Letra 1), caso contrário, -1 (Letra 2)
        if (sum >= 0) {
            return 1;  // Letra 1
        } else {
            return -1;  // Letra 2
        }
    }

    private void testRecognition() {
        int output = predict(testMatrix);

        String result;
        if (output == 1) {
            result = "A matriz TESTE parece mais com a Letra 1!";
        } else if (output == -1) {
            result = "A matriz TESTE parece mais com a Letra 2!";
        } else {
            result = "Erro no reconhecimento.";
        }

        // Exibe a resposta do teste
        JOptionPane.showMessageDialog(null, result, "Resultado do Teste", JOptionPane.INFORMATION_MESSAGE);
    }
}
