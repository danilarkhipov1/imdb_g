import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib



def load_data_from_folder(folder_path, label):
    texts = []
    labels = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)

    return texts, labels


# Путь к папкам train и test
train_dir = "C:\\Users\\user\\Downloads\\aclImdb_v1\\train"
test_dir = "C:\\Users\\user\\Downloads\\aclImdb_v1\\test"

train_texts = []
train_labels = []

test_texts = []
test_labels = []

for label in ["pos", "neg"]:
    train_folder = os.path.join(train_dir, label)
    test_folder = os.path.join(test_dir, label)

    train_text, train_label = load_data_from_folder(train_folder, label)
    test_text, test_label = load_data_from_folder(test_folder, label)

    train_texts.extend(train_text)
    test_texts.extend(test_text)

    train_labels.extend(train_label)
    test_labels.extend(test_label)

# Векторизация текстовых данных
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Обучение модели
model = MultinomialNB()
model.fit(X_train, train_labels)

# Предсказания на тестовых данных
predictions = model.predict(X_test)

# Вычисление точности
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)
# Путь к файлу model_ton.pkl
model_file = "model_ton.pkl"
# Проверяем наличие файла перед сохранением
if not os.path.exists(model_file):
    # Сохраняем модель в файл только если файл не существует
    joblib.dump(model, model_file)
model= joblib.load("model_ton.pkl")