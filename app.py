import streamlit as st
from autogluon.tabular import TabularPredictor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import os

st.set_page_config(page_title="AutoML Прогнозування", layout="wide")
st.title('Автоматизоване навчання моделей з AutoGluon')

# --- Збереження та завантаження моделі ---
def save_predictor(predictor, path='AutogluonModels/'):
    predictor.save(path)

def load_predictor(path='AutogluonModels/'):
    if os.path.exists(path):
        try:
            predictor = TabularPredictor.load(path)
            return predictor
        except Exception:
            return None
    return None

# --- Стан ---
if 'predictor' not in st.session_state:
    st.session_state['predictor'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'target_column' not in st.session_state:
    st.session_state['target_column'] = None

# --- Завантаження файлу ---
uploaded_file = st.file_uploader("Оберіть CSV файл (UTF-8)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Помилка завантаження файлу: {e}")
        st.stop()

    df = st.session_state['df']
    st.success("Файл завантажено успішно!")
    st.write("Типи колонок:")
    st.write(df.dtypes)

    # --- Вибір цільової змінної ---
    possible_targets = df.columns.tolist()
    target_column = st.selectbox("Оберіть цільову змінну (target):", possible_targets)
    st.session_state['target_column'] = target_column

    # --- Розбиття на train/test ---
    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)

    # --- Кнопка навчання ---
    if st.button('Навчити нову модель'):
        with st.spinner("Навчаємо модель..."):
            predictor = TabularPredictor(label=target_column, path='AutogluonModels/').fit(train_data)
            st.session_state['predictor'] = predictor
            save_predictor(predictor)
            st.success("Модель навчена і збережена!")

# --- Спроба завантажити наявну модель ---
if st.button("Завантажити збережену модель"):
    predictor = load_predictor('AutogluonModels/')
    if predictor:
        st.session_state['predictor'] = predictor
        st.success("Модель успішно завантажена!")
    else:
        st.error("Не вдалося завантажити модель. Перевірте наявність каталогу AutogluonModels.")

# --- Робота з навченою моделлю ---
if st.session_state['predictor'] is not None:
    predictor = st.session_state['predictor']
    df = st.session_state['df']
    target_column = st.session_state['target_column']
    
    # --- Метрики ---
    test_data = df.drop(df.sample(frac=0.8, random_state=42).index)
    y_true = test_data[target_column]
    y_pred = predictor.predict(test_data)
    acc = accuracy_score(y_true, y_pred)
    st.markdown(f"## Метрики якості моделі")
    st.write(f"**Accuracy:** {acc:.3f}")

    if set(y_true.unique()) == {0,1}:
        y_proba = predictor.predict_proba(test_data)[1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        st.write(f"**ROC-AUC:** {roc_auc:.3f}")

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'ROC крива (AUC = {roc_auc:.3f})')
        ax_roc.plot([0,1], [0,1], linestyle='--', color='grey')
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.legend()
        st.pyplot(fig_roc)
    else:
        st.info("Для ROC-AUC потрібні класи 0 і 1.")

    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # --- Прогноз ---
    st.markdown("---")
    st.subheader("Введіть значення ознак для прогнозу")
    features = [col for col in df.columns if col != target_column]
    input_dict = {}

    for feat in features:
        st.markdown(f"**{feat}**")
        if pd.api.types.is_numeric_dtype(df[feat]):
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            median_val = float(df[feat].median())
            input_val = st.slider(f"{feat}", min_value=min_val, max_value=max_val, value=median_val)
            st.markdown(f"Введене значення: `{input_val}`")
            input_dict[feat] = input_val
        else:
            options = df[feat].dropna().unique().tolist()
            input_val = st.selectbox(f"{feat}", options)
            st.markdown(f"Обране значення: `{input_val}`")
            input_dict[feat] = input_val

    input_df = pd.DataFrame([input_dict])

    if st.button('Прогнозувати'):
        pred = predictor.predict(input_df)[0]
        st.write(f"### Прогноз: `{pred}`")

else:
    st.info("Завантажте датасет, оберіть цільову змінну та навчіть або завантажте модель.")

# --- Інфо ---
st.sidebar.title("Інструкція")
st.sidebar.write("""
1. Завантажте CSV-файл (UTF-8)
2. Оберіть колонку цільової змінної
3. Навчіть модель або завантажте збережену
4. Перегляньте метрики якості
5. Введіть дані для прогнозу
6. Отримайте результат
""")
st.sidebar.markdown("---")

