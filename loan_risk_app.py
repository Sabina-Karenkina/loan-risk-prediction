import streamlit as st
import pandas as pd
import pickle

#Конфигурация страницы 
st.set_page_config(
    page_title='Одобрение кредита',
    page_icon='🏦',
    layout='wide'
)

# Загрузка модели (кэшируется — загружается один раз!)
@st.cache_resource
def load_model():
    with open('loan_risk_model.joblib', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Заголовок 
st.title('🏦 Система проверки одобрения кредита')
st.markdown('''
Введите данные заемщика и получите мгновенное решение по кредиту.
*Демо-модель на основе XGBoost | Казахстан 🇰🇿*
''')
st.divider()

#Форма ввода в 2 колонки
col_left, col_right = st.columns(2)

with col_left:
    st.subheader('👤 Личные данные')
    City = st.selectbox(
        'Город',
        ['Chicago', 'San Francisco', 'Houston', 'New York']
    )
    Education = st.selectbox(
        'Образование',
        ['High School', 'Bachelors', 'Masters','PhD']
    )
    Age = st.number_input(
        'Возраст', min_value=18, max_value=69, value=43, step=1
    )
    Gender = st.selectbox(
        'Пол',
        ['Female','Male']
    )


with col_right:
    st.subheader('💰 Финансовые данные')
    Income = st.number_input(
        '💵 Ежемесячный доход',
        min_value=0, max_value=100000,
        value=50000, step=1000
    )
    LoanAmount = st.number_input(
        "🏧 Сумма кредита",
        min_value=0, max_value=50000,
        value=20000, step=1000
    )
    YearsExperience = st.slider(
        "📅 Опыт работы (лет)",
        min_value=0, max_value=40, value=20
    )
    EmploymentType = st.selectbox(
        'Тип занятости',
        ['Self-Employed', 'Unemployed', 'Salaried']
    )
    CreditScore = st.slider(
        'Кредитный рейтинг',
        min_value=300,
        max_value=850,
        value=575
    )

st.divider()

# Кнопка предсказания 
if st.button('🔍 Проверить клиента', type='primary', use_container_width=True):

    # Формируем DataFrame для модели
    input_data = pd.DataFrame([{
        'Age': Age,
        'Income': Income,
        'LoanAmount': LoanAmount,
        'CreditScore': CreditScore,
        'YearsExperience': YearsExperience,
        'Gender': Gender,
        'Education': Education,
        'City': City,
        'EmploymentType': EmploymentType
    }])

    # Предсказание
    prediction    = model.predict(input_data)[0]
    probability   = model.predict_proba(input_data)[0][1]

    # Результат 
    if prediction == 1:
        st.success('✅ КРЕДИТ ОДОБРЕН')
    else:
        st.error('❌ КРЕДИТ ОТКЛОНЁН')

    #  Метрики 
    m1, m2, m3 = st.columns(3)
    m1.metric(
        '🎯 Вероятность одобрения',
        f"{probability:.1%}"
    )
    m2.metric(
        '💳 Сумма кредита',
        f"{LoanAmount:,.0f} "
    )
    m3.metric(
        '💵 Ежемесячный доход',
        f"{Income:,.0f}"
    )

    # Прогресс-бар нагрузки
    st.subheader('📊 Кредитная нагрузка')
    # Ориентировочный платёж (3 года, ~12%)
    monthly_payment = LoanAmount * 0.033
    load_ratio = monthly_payment / Income
    st.progress(min(load_ratio, 1.0))
    st.write(f"Ежемесячный платёж: ~{monthly_payment:,.0f} ({load_ratio:.1%} от дохода)")

    if load_ratio > 0.6:
        st.warning('⚠️ Высокая долговая нагрузка (более 60% дохода)')
    elif load_ratio > 0.4:
        st.info('ℹ️ Умеренная нагрузка (40–60% дохода)')
    else:
        st.success('✅ Комфортная нагрузка (менее 40% дохода)')

    # Предупреждения (hard_reject логика) 
    if EmploymentType == 'Unemployed':
        st.error("🚫 Автоматический отказ: статус 'Безработный'")
    if Income / LoanAmount < 0.05:
        st.error('🚫 Автоматический отказ: доход < 5% от суммы кредита')

    #  Детали решения 
    with st.expander('📋 Детали введённых данных'):
        display_df = input_data.T.reset_index()
        display_df.columns = ['Параметр', 'Значение']
        display_df['Значение'] = display_df['Значение'].astype(str)
        st.dataframe(display_df, width='stretch')

#  Подвал
st.caption('🏦 Кредитный скоринг | XGBoost Pipeline | Курс ML, Неделя 19 | 🇰🇿')
