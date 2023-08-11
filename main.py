import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import model_ton
import model_rating
import joblib

vectorizer = joblib.load("vectorizer.pkl")
# Заголовок веб-приложения
st.title("Определение тональности и рейтинга отзыва")

# Поле для ввода текстового отзыва
review_text = st.text_area("Введите текст отзыва (только на английском языке):")

# Функция для обработки события "Предсказать тональность"
def predict_sentiment():
    review_vector = vectorizer.transform([review_text])
    prediction_ton = model_ton.model.predict(review_vector)[0]
    st.write("Предсказанная тональность:", prediction_ton)

# Функция для обработки события "Предсказать рейтинг фильма"
def predict_rating():
    review_vector_rt = vectorizer.transform([review_text])
    prediction_rating = model_rating.model.predict(review_vector_rt)[0]
    # Округление рейтинга до одной десятой
    rounded_rating = round(prediction_rating, 0)
    st.write("Предсказанный рейтинг фильма:", rounded_rating)

# Кнопка для предсказания тональности
if st.button("Предсказать тональность"):
    if review_text:
        predict_sentiment()
    else:
        st.write("Пожалуйста, введите текст отзыва.")

# Кнопка для предсказания рейтинга
if st.button("Предсказать рейтинг фильма"):
    if review_text:
        predict_rating()
    else:
        st.write("Пожалуйста, введите текст отзыва.")
