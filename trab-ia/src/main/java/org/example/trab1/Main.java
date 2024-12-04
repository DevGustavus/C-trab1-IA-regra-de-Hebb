package org.example.trab1;

import javax.swing.*;
import java.awt.*;

// ================================= HEBB =================================
public class Main {

    private static final int SIZE = 10; // Tamanho da matriz (10x10)
    private int[][] firstLetterMatrix = new int[SIZE][SIZE];
    private int[][] secondLetterMatrix = new int[SIZE][SIZE];
    private int[][] testMatrix = new int[SIZE][SIZE];

    public static void main(String[] args) {
        SwingUtilities.invokeLater(Main::new);
    }

    public Main() {
        JFrame frame = new JFrame("Regra de Hebb - Reconhecimento de Letras");
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
        JButton testButton = new JButton("Testar");
        testButton.addActionListener(e -> testRecognition());
        buttonPanel.add(testButton);

        frame.add(mainPanel, BorderLayout.CENTER);
        frame.add(buttonPanel, BorderLayout.SOUTH);

        frame.setVisible(true);
        frame.setResizable(false);
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

    private void testRecognition() {
        // Calcula as correspondências usando pesos ponderados
        double firstLetterSimilarity = calculateSimilarity(firstLetterMatrix, testMatrix);
        double secondLetterSimilarity = calculateSimilarity(secondLetterMatrix, testMatrix);

        // Determina qual letra é mais semelhante
        String result;
        if (firstLetterSimilarity > secondLetterSimilarity) {
            result = "A matriz TESTE parece mais com a Letra 1!";
        } else if (secondLetterSimilarity > firstLetterSimilarity) {
            result = "A matriz TESTE parece mais com a Letra 2!";
        } else {
            result = "Não foi possível determinar uma correspondência clara.";
        }

        JOptionPane.showMessageDialog(null, result, "Resultado do Teste", JOptionPane.INFORMATION_MESSAGE);
    }

    private double calculateSimilarity(int[][] letterMatrix, int[][] testMatrix) {
        double similarity = 0.0;
        double totalActivePixels = 0.0; // Conta pixels ativos no teste

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (testMatrix[i][j] == 1) {
                    totalActivePixels++;
                    if (letterMatrix[i][j] == 1) {
                        similarity += 1.0; // Incrementa se o pixel for coincidente
                    } else {
                        similarity -= 0.5; // Penalidade por pixel divergente
                    }
                }
            }
        }

        // Normaliza a pontuação pela quantidade de pixels ativos no teste
        return totalActivePixels > 0 ? similarity / totalActivePixels : 0.0;
    }
}
