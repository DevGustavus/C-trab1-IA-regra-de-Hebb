package org.example.trab2;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Main_InterfaceGrafica extends JFrame {
    private static final int GRID_SIZE = 10; // Tamanho da grade 10x10
    private JButton[][] grid1, grid2, grid3; // As grades para desenhar as letras
    private JButton testButton, trainButton; // Botões para testar e treinar a rede
    private JPanel panel1, panel2, panel3, panelTest; // Painéis para cada grade
    private JLabel label1, label2, labelTest; // Labels para os cabeçalhos

    // Redes e variáveis do perceptron
    private float[][] entradas; // Entradas (100 valores)
    private int[] target = new int[2]; // Saídas desejadas para "A" e "B"
    private float[] w = new float[100]; // Pesos para as entradas
    private float b = 0.0f; // Viés
    private float alfa = 0.01f; // Taxa de aprendizagem
    private float limiar = 0.0f; // Limiar para a ativação
    private float yTeste; // Saída para teste

    public Main_InterfaceGrafica() {
        // Inicializando a interface gráfica
        setTitle("Perceptron - Desenho de Letras");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));

        // Inicializando a rede
        target[0] = -1; // Reconhecimento de A
        target[1] = 1;  // Reconhecimento de B
        for (int i = 0; i < 100; i++) {
            w[i] = 0.5f; // Inicializando pesos com valores arbitrários
        }

        // Inicializando os painéis de desenho com cabeçalhos
        panel1 = new JPanel();
        panel2 = new JPanel();
        panel3 = new JPanel();
        panelTest = new JPanel();

        // Labels para os cabeçalhos
        label1 = new JLabel("Letra 1");
        label2 = new JLabel("Letra 2");
        labelTest = new JLabel("Teste");

        // Configuração do layout para os painéis
        panel1.setLayout(new BoxLayout(panel1, BoxLayout.Y_AXIS));
        panel2.setLayout(new BoxLayout(panel2, BoxLayout.Y_AXIS));
        panel3.setLayout(new BoxLayout(panel3, BoxLayout.Y_AXIS));
        panelTest.setLayout(new FlowLayout());

        // Adicionando cabeçalhos e grades para desenhar
        panel1.add(label1);
        grid1 = new JButton[GRID_SIZE][GRID_SIZE];
        createGrid(panel1, grid1);

        panel2.add(label2);
        grid2 = new JButton[GRID_SIZE][GRID_SIZE];
        createGrid(panel2, grid2);

        panel3.add(labelTest);
        grid3 = new JButton[GRID_SIZE][GRID_SIZE];
        createGrid(panel3, grid3);

        // Adiciona os painéis com cabeçalhos
        add(panel1);
        add(panel2);
        add(panel3);

        // Adiciona o painel de teste com o botão
        testButton = new JButton("Testar Letra");
        testButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Processa a entrada dos campos
                float[] input1 = getGridInput(grid1);
                float[] input2 = getGridInput(grid2);
                float[] input3 = getGridInput(grid3);

                // Testar a rede com os dados desenhados
                testNetwork(input1);
                testNetwork(input2);
                testNetwork(input3);
            }
        });

        // Adiciona o painel de treinamento com o botão
        trainButton = new JButton("Treinar Perceptron");
        trainButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Treinar a rede com as entradas e saídas desejadas
                trainNetwork();
            }
        });

        panelTest.add(testButton);
        panelTest.add(trainButton);
        add(panelTest);
    }

    // Cria a grade de botões para o campo de desenho
    private void createGrid(JPanel panel, JButton[][] grid) {
        JPanel gridPanel = new JPanel(new GridLayout(GRID_SIZE, GRID_SIZE));
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                grid[i][j] = new JButton();
                grid[i][j].setPreferredSize(new Dimension(30, 30));
                grid[i][j].setBackground(Color.WHITE);
                grid[i][j].addActionListener(new ActionListener() {
                    @Override
                    public void actionPerformed(ActionEvent e) {
                        JButton btn = (JButton) e.getSource();
                        // Altera a cor para simular um desenho
                        if (btn.getBackground() == Color.WHITE) {
                            btn.setBackground(Color.BLACK);
                        } else {
                            btn.setBackground(Color.WHITE);
                        }
                    }
                });
                gridPanel.add(grid[i][j]);
            }
        }
        panel.add(gridPanel);
    }

    // Converte a entrada de uma grade de desenho em um vetor de 100 valores (-1 ou 1)
    private float[] getGridInput(JButton[][] grid) {
        float[] input = new float[100];
        int k = 0;
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                if (grid[i][j].getBackground() == Color.BLACK) {
                    input[k] = 1; // Se o botão for preto, consideramos como 1
                } else {
                    input[k] = -1; // Se for branco, consideramos como -1
                }
                k++;
            }
        }
        return input;
    }

    // Testa a rede com as entradas fornecidas
    private void testNetwork(float[] input) {
        float resultado = 0;
        for (int i = 0; i < 100; i++) {
            resultado += input[i] * w[i];
        }
        resultado += b;

        // Verifica se a saída é maior ou igual ao limiar
        yTeste = (resultado >= limiar) ? 1 : -1;

        System.out.printf("\nSaída da rede: %.2f\n", yTeste);
    }

    // Função para treinar o perceptron
    private void trainNetwork() {
        // Exemplos de entrada para as letras A e B
        entradas = new float[][]{
                getGridInput(grid1), // Letra 1 (A)
                getGridInput(grid2)  // Letra 2 (B)
        };

        // Treinamento do perceptron
        for (int ciclo = 0; ciclo < 1000; ciclo++) { // Número de ciclos de treinamento
            for (int i = 0; i < entradas.length; i++) {
                float[] input = entradas[i];
                float resultado = 0;

                // Calcular a saída
                for (int j = 0; j < 100; j++) {
                    resultado += input[j] * w[j];
                }
                resultado += b;
                float yLiq = (resultado >= limiar) ? 1 : -1;

                // Cálculo do erro
                int erro = target[i] - (int) yLiq;

                // Atualiza os pesos e viés
                for (int j = 0; j < 100; j++) {
                    w[j] += alfa * erro * input[j];
                }
                b += alfa * erro;
            }
        }
        System.out.println("Treinamento concluído.");
    }

    public static void main(String[] args) {
        Main_InterfaceGrafica perceptron = new Main_InterfaceGrafica();
        perceptron.setVisible(true);
    }
}
