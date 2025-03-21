import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data import numbers

# Загружаем данные MNIST
with open("./mnist_test.csv", 'r') as f:
    data_list = f.readlines()

allowed_digits = numbers

data_by_digit = {i: [] for i in allowed_digits}
for record in data_list:
    all_values = record.strip().split(',')
    label = int(all_values[0])
    if label in allowed_digits:
        data_by_digit[label].append(record)




# Формируем обучающую и тестовую выборки
train_data, test_data = [], []
for digit in allowed_digits:
    train_data.extend(data_by_digit[digit][:200])  # 200 на обучение
    test_data.extend(data_by_digit[digit][200:300])  # 100 на тест

# Подготовка данных
X_train = np.array([np.array(record.split(',')[1:], dtype=float).reshape(28, 28, 1) / 255.0 for record in train_data])
y_train = np.array([int(record.split(',')[0]) for record in train_data])
X_test = np.array([np.array(record.split(',')[1:], dtype=float).reshape(28, 28, 1) / 255.0 for record in test_data])
y_test = np.array([int(record.split(',')[0]) for record in test_data])

# Преобразуем метки в one-hot encoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical([list(allowed_digits).index(label) for label in y_train], len(allowed_digits))
y_test = to_categorical([list(allowed_digits).index(label) for label in y_test], len(allowed_digits))

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(allowed_digits), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Графики потерь
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Потери при обучении')
plt.plot(history.history['val_loss'], label='Тестовые потери')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.title('График функции потерь')
plt.show()

# Графики точности
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Обучающая точность')
plt.plot(history.history['val_accuracy'], label='Тестовая точность')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.title('График точности')
plt.show()

# Вычисление точности
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Точность модели на тестовых данных: {accuracy * 100:.2f}%")

# Матрица ошибок
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=allowed_digits, yticklabels=allowed_digits)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок')
plt.show()

# Проверка последнего изображения из тренировочной выборки
last_image = X_train[-1].reshape(1, 28, 28, 1)
prediction = model.predict(last_image)
predicted_label = list(allowed_digits)[np.argmax(prediction)]
plt.imshow(X_train[-1].reshape(28, 28), cmap='Greys')
correct_label_index = np.argmax(y_train[-1])  # Индекс метки в one-hot encoding
correct_label = list(allowed_digits)[correct_label_index]  # Преобразуем индекс в саму цифру
print(f"Правильный ответ: {correct_label}")
print(f"Предсказание сети: {predicted_label}")
plt.show()
