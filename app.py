import streamlit as st
from openai import OpenAI
import json
import pandas as pd
from PIL import Image
import io
import base64

# Настройка страницы
st.set_page_config(page_title="ЕГЭ-Эксперт", page_icon="🎓", layout="wide")

# Инициализация клиента
try:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
except:
    st.error("Ошибка: API ключ не найден в Secrets. Проверьте настройки Streamlit Cloud.")
    st.stop()

def get_base64_image(image_file):
    image = Image.open(image_file)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти к", ["Главная", "Проверка"])

if page == "Главная":
    st.title("🎓 ЕГЭ-Эксперт")
    st.markdown("Сервис для автоматической проверки сочинений по критериям ФИПИ.")
else:
    st.title("✍️ Проверка сочинения")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        source_text = st.text_area("Исходный текст:", height=100)
        input_type = st.radio("Источник сочинения:", ["Текст", "Фото рукописи"])
        if input_type == "Текст":
            essay_text = st.text_area("Текст сочинения:", height=200)
        else:
            uploaded_image = st.file_uploader("Загрузите фото...", type=["jpg", "png"])
            essay_text = None

    with col2:
        if st.button("🚀 Проверить"):
            with st.spinner("Анализирую..."):
                prompt = f"""
                Ты — строгий эксперт ЕГЭ. Исходный текст: {source_text}. 
                Проанализируй сочинение. Верни JSON с полями:
                "corrected_text" (текст с выделенными ошибками **...**),
                "table" (список словарей [{"Критерий": "К1", "Баллы": 0}, ...]),
                "details" (словарь {"К1": "разбор...", ...}),
                "total" (итоговый балл).
                Отвечай ТОЛЬКО JSON-ом.
                """
                
                messages = [{"role": "user", "content": prompt + (f" Сочинение: {essay_text}" if input_type=="Текст" else "")}]
                if input_type == "Фото рукописи":
                    messages[0]["content"] = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(uploaded_image)}"}}]

                try:
                    response = client.chat.completions.create(model="google/gemma-4-31b-it:free", messages=messages)
                    data = json.loads(response.choices[0].message.content.replace('```json', '').replace('```', ''))

                    st.metric("Итого баллов", data.get("total", 0))
                    st.info(data.get("corrected_text", "Ошибка анализа текста"))
                    
                    st.subheader("📊 Разбор по критериям")
                    table_data = data.get("table", [])
                    details = data.get("details", {})
                    
                    for k in ["К1", "К2", "К3", "К4", "К5", "К6", "К7", "К8", "К9", "К10", "К11", "К12"]:
                        bal = next((item.get("Баллы", 0) for item in table_data if item.get("Критерий") == k), 0)
                        with st.expander(f"{k} — {bal} баллов"):
                            st.write(details.get(k, "Комментариев нет."))
                except Exception as e:
                    st.error(f"Ошибка парсинга: {e}")
