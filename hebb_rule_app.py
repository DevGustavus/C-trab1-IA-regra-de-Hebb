import numpy as np
import tkinter as tk

class HebbLearningApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Hebb Learning - A vs B")

        # Inicializar pesos para letras A e B
        self.weights_A = np.zeros((10, 10))
        self.weights_B = np.zeros((10, 10))
        
        # Grade de entrada (onde usuário desenha)
        self.inputs = np.zeros((10, 10), dtype=int)
        self.grid_buttons = []

        # Configurar grid de botões
        for i in range(10):
            row = []
            for j in range(10):
                btn = tk.Button(master, width=2, height=1, command=lambda x=i, y=j: self.toggle_pixel(x, y))
                btn.grid(row=i, column=j)
                row.append(btn)
            self.grid_buttons.append(row)

        # Botões de Treinamento, Reconhecimento e Reset
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
        # Alterar pixel de entrada
        self.inputs[x, y] = 1 if self.inputs[x, y] == 0 else 0
        color = "black" if self.inputs[x, y] == 1 else "white"
        self.grid_buttons[x][y].configure(bg=color)

    def train_with_A(self):
        self.weights_A += self.inputs
        self.clear_inputs()
        # Habilitar o botão para treinar com B após o treinamento com A
        self.train_B_button.config(state="normal")
        self.train_A_button.config(state="disabled")  # Desabilitar o botão Treinar com A

    def train_with_B(self):
        self.weights_B += self.inputs
        self.clear_inputs()
        # Habilitar o botão para reconhecer após o treinamento com B
        self.recognize_button.config(state="normal")
        self.train_B_button.config(state="disabled")  # Desabilitar o botão Treinar com B

    def recognize(self):
        score_A = np.sum(self.inputs * self.weights_A)
        score_B = np.sum(self.inputs * self.weights_B)
        result = "A" if score_A > score_B else "B"
        self.result_label.config(text=f"Resultado: {result}")

    def clear_inputs(self):
        # Redefinir todos os valores de entrada para zero e mudar a cor dos botões para branco
        self.inputs.fill(0)
        for i in range(10):
            for j in range(10):
                self.grid_buttons[i][j].configure(bg="white")

    def reset_all(self):
        # Limpar pesos e entradas e resetar o estado dos botões
        self.weights_A.fill(0)
        self.weights_B.fill(0)
        self.clear_inputs()
        
        # Resetar o estado inicial dos botões
        self.train_A_button.config(state="normal")
        self.train_B_button.config(state="disabled")
        self.recognize_button.config(state="disabled")
        
        # Limpar o resultado
        self.result_label.config(text="Resultado: ")

# Executar a aplicação
root = tk.Tk()
app = HebbLearningApp(root)
root.mainloop()
