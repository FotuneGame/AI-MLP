import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data import numbers

def showData(data, x=6, y=10):
    if not data:
        print("Данные пусты.")
        return


    a = min(int(len(data) / len(numbers)), x)
    if a <= 0:
        a = 1
    b = min(int(len(data) / a), y)
    if b <= 0:
        b = 1


    fig, axs = plt.subplots(a, b, figsize=(18, 10), constrained_layout=True)
    
    # Если a или b равно 1, axs будет не двумерным массивом
    if a == 1 and b == 1:
        axs = np.array([[axs]])
    elif a == 1:
        axs = axs.reshape(1, -1)
    elif b == 1:
        axs = axs.reshape(-1, 1)

    num = 0
    for i in range(a):
        for j in range(b):
            image_array = np.asarray(data[num][1:], dtype='int64').reshape((28, 28))
            axs[i, j].imshow(image_array, cmap='Greys', interpolation='none')
            num += 1

    plt.show()


def showGraphError(error_epochs, title):
    plt.plot(error_epochs)
    plt.xlabel('Эпохи')
    plt.ylabel('Средняя ошибка')
    plt.title(title)
    plt.grid(True)
    plt.show()


def showGraphAccess(error_epochs,title):
    access = np.zeros(len(error_epochs))
    for i in range(0,len(error_epochs)):
        access[i] = 1 - error_epochs[i]
    plt.plot(access)
    plt.xlabel('Эпохи')
    plt.ylabel('Точноть распознования')
    plt.title(title)
    plt.grid(True)
    plt.show()



def showConfusionMatrix(true_labels, predicted_labels):
    # Создание матрицы неточностей
    cm = confusion_matrix(true_labels, predicted_labels, labels=numbers)

    # Визуализация матрицы
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=numbers, yticklabels=numbers)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return cm