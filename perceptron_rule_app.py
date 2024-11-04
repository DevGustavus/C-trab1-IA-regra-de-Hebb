import numpy as np
import tkinter as tk

class PerceptronApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Perceptron - Reconhecimento A vs B")

        # Inicializar pesos e bias
        self.pesos = np.zeros(100)
        self.bias = 0
        self.entrada_atual = np.zeros(100)  # Entrada 10x10 achatada

        # Grid de entrada para desenhar
        self.grid_buttons = []
        for i in range(10):
            row = []
            for j in range(10):
                btn = tk.Button(master, width=2, height=1, command=lambda x=i, y=j: self.toggle_pixel(x, y))
                btn.grid(row=i, column=j)
                row.append(btn)
            self.grid_buttons.append(row)

        # Botões de controle
        self.train_A_button = tk.Button(master, text="Treinar com A", command=self.train_with_A)
        self.train_A_button.grid(row=11, column=0, columnspan=3)

        self.train_B_button = tk.Button(master, text="Treinar com B", command=self.train_with_B, state="disabled")
        self.train_B_button.grid(row=11, column=3, columnspan=3)

        self.recognize_button = tk.Button(master, text="Reconhecer", command=self.recognize, state="disabled")
        self.recognize_button.grid(row=11, column=6, columnspan=4)

        self.reset_button = tk.Button(master, text="Resetar", command=self.reset_all)
        self.reset_button.grid(row=12, column=0, columnspan=10)

        self.result_label = tk.Label(master, text="Resultado: ")
        self.result_label.grid(row=13, column=0, columnspan=10)

    def toggle_pixel(self, x, y):
        # Alternar o estado do pixel e atualizar cor do botão
        index = x * 10 + y
        self.entrada_atual[index] = 1 if self.entrada_atual[index] == 0 else 0
        color = "black" if self.entrada_atual[index] == 1 else "white"
        self.grid_buttons[x][y].configure(bg=color)

    def train_with_A(self):
        # Treinamento com a letra A e atualiza botões
        self.treinar(1)
        self.train_A_button.config(state="disabled")
        self.train_B_button.config(state="normal")
        self.recognize_button.config(state="disabled")

    def train_with_B(self):
        # Treinamento com a letra B e atualiza botões
        self.treinar(-1)
        self.train_B_button.config(state="disabled")
        self.recognize_button.config(state="normal")

    def treinar(self, alvo):
        # Função de treinamento do Perceptron
        y = self.ativacao(np.dot(self.pesos, self.entrada_atual) + self.bias)
        erro = alvo - y
        self.pesos += erro * self.entrada_atual
        self.bias += erro
        self.clear_inputs()

    def ativacao(self, x):
        # Função de ativação
        return 1 if x >= 0 else -1

    def recognize(self):
        # Função para reconhecer a entrada atual
        resultado = self.ativacao(np.dot(self.pesos, self.entrada_atual) + self.bias)
        self.result_label.config(text=f"Resultado: {'A' if resultado == 1 else 'B'}")

    def clear_inputs(self):
        # Limpar a entrada e redefinir cor dos botões para branco
        self.entrada_atual.fill(0)
        for i in range(10):
            for j in range(10):
                self.grid_buttons[i][j].configure(bg="white")

    def reset_all(self):
        # Resetar pesos, entradas e botões ao estado inicial
        self.pesos.fill(0)
        self.bias = 0
        self.clear_inputs()
        self.train_A_button.config(state="normal")
        self.train_B_button.config(state="disabled")
        self.recognize_button.config(state="disabled")
        self.result_label.config(text="Resultado: ")

# Executar a aplicação
root = tk.Tk()
app = PerceptronApp(root)
root.mainloop()
