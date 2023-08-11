import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


def load_data_from_folder(folder_path):
    texts = []
    ratings = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                # Извлекаем рейтинг из имени файла с помощью регулярного выражения
                rating = re.search(r"(\d{1,2})\.txt", file).group(1)
                ratings.append(int(rating))

    return texts, ratings


# Путь к папкам train и test
train_dir = "C:\\Users\\user\\Downloads\\aclImdb_v1\\train"
test_dir = "C:\\Users\\user\\Downloads\\aclImdb_v1\\test"

train_texts = []
train_ratings = []

test_texts = []
test_ratings = []

for label in ["pos", "neg"]:
    train_folder = os.path.join(train_dir, label)
    test_folder = os.path.join(test_dir, label)

    train_text, train_rating = load_data_from_folder(train_folder)
    test_text, test_rating = load_data_from_folder(test_folder)

    train_texts.extend(train_text)
    train_ratings.extend(train_rating)

    test_texts.extend(test_text)
    test_ratings.extend(test_rating)

# Векторизация текстовых данных
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, train_ratings)

# Предсказания рейтинга на тестовых данных
predictions = model.predict(X_test)

# Вычисление средней квадратичной ошибки
mse = mean_squared_error(test_ratings, predictions)
print("Mean Squared Error:", mse)
# Путь к файлу model_rt.pkl
model_file = "model_rating.pkl"
joblib.dump(model, model_file)
vectorizer_file="vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_file)
    # Сохраняем модель в файл только если файл не существует
model = joblib.load("model_rating.pkl")