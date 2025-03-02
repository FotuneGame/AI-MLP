import numpy as np
from data import numbers


class MLP:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Инициализация весов и смещений
        self.weights_input_hidden = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.weights_hidden_output = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        self.bias_hidden = np.random.rand(self.hidden_nodes, 1) - 0.5
        self.bias_output = np.random.rand(self.output_nodes, 1) - 0.5

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        # Прямое распространение (feedforward)
        hidden_inputs = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)

        # Ошибка на выходном слое
        output_errors = targets - final_outputs

        # Ошибка на скрытом слое
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        # Обновление весов и смещений (обратное распространение)
        self.weights_hidden_output += self.learning_rate * np.dot((output_errors * self.sigmoid_derivative(final_outputs)), hidden_outputs.T)
        self.bias_output += self.learning_rate * (output_errors * self.sigmoid_derivative(final_outputs))

        self.weights_input_hidden += self.learning_rate * np.dot((hidden_errors * self.sigmoid_derivative(hidden_outputs)), inputs.T)
        self.bias_hidden += self.learning_rate * (hidden_errors * self.sigmoid_derivative(hidden_outputs))

        # Вероятность
        output_chances = final_outputs.flatten()
        return [output_errors, output_chances]


# Параметры
input_nodes = 784  # Для MNIST
hidden_nodes = len(numbers)  # Количество нейронов в скрытом слое
output_nodes = len(numbers)  # Количество выходных нейронов (по числу классов)
learning_rate = 0.1
mlp = MLP(input_nodes, hidden_nodes, output_nodes, learning_rate)


def learnMLP(data, epochs):
    error_epochs = []
    number = 0
    chance = float('-inf')

    for e in range(epochs):
        sum_error = 0
        for record in data:
            inputs = (np.asarray(record[1:], dtype=float) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(numbers.index(int(record[0])))] = 0.99
            errors,chances = mlp.train(inputs, targets)
            sum_error += np.linalg.norm(errors)

            maxIndex = np.argmax(chances)  # Индекс максимальной вероятности
            number = numbers[maxIndex]
            chance = chances[maxIndex]

        error_epochs.append(sum_error / len(data))  # Средняя ошибка на эпоху
        print("Epocha now: ",e, "Last is: ",epochs)
    return [error_epochs, number, chance ]