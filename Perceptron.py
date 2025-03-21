import numpy as np
from data import numbers



class Perceptron:
    def __init__(self, input_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        # Инициализация весов случайными значениями
        self.weights = np.random.rand(self.output_nodes, self.input_nodes) - 0.5

    def train(self, inputs, targets, isTest=False):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        # Прямое распространение
        outputs = np.dot(self.weights, inputs)
        outputs = 1 / (1 + np.exp(-outputs))  # Сигмоидальная функция активации

        # Ошибка
        output_errors = targets - outputs

        # Обновление весов
        if(not(isTest)):
            self.weights += self.learning_rate * np.dot((output_errors * outputs * (1 - outputs)), inputs.T)

        # Вероятность
        output_chances = outputs.flatten()
        return [output_errors, output_chances]

# Параметры
input_nodes = 784  # Для MNIST
output_nodes = len(numbers) #необходимое число цифр для анализа
learning_rate = 0.1
n = Perceptron(input_nodes, output_nodes, learning_rate)



def learnP(data,test_data,epochs):
    error_epochs = []
    number = 0
    chance = float('-inf')
    error_test_epochs = []
    number_test = 0
    chance_test = float('-inf')
    for e in range(epochs):
        sum_error = 0
        sum_error_test = 0
        for record in data:
            inputs = (np.asarray(record[1:], dtype=float) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(numbers.index(int(record[0])))] = 0.99
            errors, chances = n.train(inputs, targets)
    
            sum_error += np.linalg.norm(errors)

            maxIndex = np.argmax(chances)  # Индекс максимальной вероятности
            number = numbers[maxIndex]
            chance = chances[maxIndex]
        
        for record in test_data:
            inputs = (np.asarray(record[1:], dtype=float) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(numbers.index(int(record[0])))] = 0.99
            error_test, chances = n.train(inputs, targets, True)
    
            sum_error_test += np.linalg.norm(error_test)

            maxIndex = np.argmax(chances)  # Индекс максимальной вероятности
            number_test = numbers[maxIndex]
            chance_test = chances[maxIndex]

        if(len(data)):
            error_epochs.append(sum_error / len(data))  # Средняя ошибка на эпоху
        if(len(test_data)):
            error_test_epochs.append(sum_error_test / len(test_data)) # Средняя ошибка на эпоху для тестового данного
        print("Epocha now: ",e, "Last is: ",epochs)
    return [error_epochs, number, chance,error_test_epochs, number_test, chance_test]
