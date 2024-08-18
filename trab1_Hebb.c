#include <stdio.h>
#include <conio.h>

void printLetter(float entrada[64])
{
    for (int i = 0; i < 8; i++)
    { // 8 linhas
        for (int j = 0; j < 8; j++)
        { // 8 colunas
            if (entrada[i * 8 + j] == 1)
            {
                printf("# "); // Ponto ativo
            }
            else
            {
                printf(". "); // Espaço vazio
            }
        }
        printf("\n"); // Quebra de linha ao final de cada linha
    }
    printf("\n"); // Quebra de linha após a letra
}

int main()
{
    float entrada[2][64]; // 64 valores e dois padrões (A e B).
    float y[] = {1, -1};  // O 1 ser o A e -1 ser o B.
    float deltaW[64], deltaB, w[64], b, deltaTeste;
    int cont1, cont2, teste[2];
    printf("\nPrograma Regra de Hebb para reconhecimento de A e B");
    for (cont2 = 0; cont2 < 64; cont2++)
    {
        w[cont2] = 0;
        entrada[0][cont2] = -1;
        entrada[1][cont2] = -1;
    }
    b = 0;

    // Inser o de dados da letra A (entrada).
    entrada[0][3] = entrada[0][10] = entrada[0][12] = entrada[0][18] = entrada[0][20] = 1;
    entrada[0][25] = entrada[0][29] = entrada[0][33] = entrada[0][34] = entrada[0][35] = 1;
    entrada[0][36] = entrada[0][37] = entrada[0][41] = entrada[0][45] = entrada[0][49] = 1;
    entrada[0][53] = entrada[0][57] = entrada[0][61] = 1;
    // Inser o de dados da letra B(entrada).
    entrada[1][9] = entrada[1][10] = entrada[1][11] = entrada[1][17] = entrada[1][19] = 1;
    entrada[1][25] = entrada[1][26] = entrada[1][33] = entrada[1][34] = entrada[1][11] = 1;
    entrada[1][43] = entrada[1][49] = entrada[1][51] = entrada[1][57] = entrada[1][58] = 1;
    entrada[1][59] = 1;

    // Imprime a letra A
    printf("\n\nLetra A:\n");
    printLetter(entrada[0]);

    // Imprime a letra B
    printf("Letra B:\n");
    printLetter(entrada[1]);

    // Aplica o da regra.
    for (cont1 = 0; cont1 < 2; cont1++)
    {
        for (cont2 = 0; cont2 < 64; cont2++)
        { // Os valores de entrada (A e B) serão mulƟplicados pela saída correspondente.
            deltaW[cont2] = entrada[cont1][cont2] * y[cont1];
        }
        deltaB = y[cont1];
        for (cont2 = 0; cont2 < 64; cont2++)
        { // Cria o dos pesos.
            // w[cont2] = w[cont2] + deltaW[cont2];
            w[cont2] = w[cont2] + (entrada[cont1][cont2] * y[cont1]);
        }
        b = b + deltaB;
    }

    // Teste: os pesos e o bias são importantes.
    for (cont1 = 0; cont1 < 2; cont1++)
    {
        deltaTeste = 0; // Somat rio.
        for (cont2 = 0; cont2 < 64; cont2++)
        {
            deltaTeste = deltaTeste + (w[cont2] * entrada[cont1][cont2]);
        }
        deltaTeste = deltaTeste + b;
        if (deltaTeste >= 0)
        {
            teste[cont1] = 1; // Isso um A
        }
        else
        {
            teste[cont1] = -1; // Isso um B.
        }
    }

    printf("\n Saida esperada: 1(A) -1(B)");
    printf("\n Saida encontrada: %i %i \n\n", teste[0], teste[1]);
    getch();
}