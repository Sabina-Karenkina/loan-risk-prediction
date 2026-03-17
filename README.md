#Loan Risk Prediction App

Приложение для оценки вероятности одобрения кредита с использованием моделей машинного обучения.
Создано на основе датасета:
https://www.kaggle.com/datasets/sohailkhan05/loan-risk-prediction/data
##  Используемые модели
- XGBoost

##  Что сделано
- Проведен EDA (анализ данных)
- Обработаны пропуски и отрицательные значения
- Учтён дисбаланс классов
- Выполнен подбор гиперпараметров
- Реализовано веб-приложение на Streamlit

##  Demo
https://loan-risk-prediction-dpfnhkr2nsey6f9asce7bm.streamlit.app/

##  Технологии
- Python
- scikit-learn
- XGBoost
- Streamlit

##  Структура проекта
- `loan_risk_app.py` — приложение
- loan_risk_model.joblib` — обученная модель
- `requirements.txt` — зависимости
