import streamlit as st
from openai import OpenAI
import json
import pandas as pd
from PIL import Image
import io
import base64
import re

# Настройка страницы
st.set_page_config(page_title="ЕГЭ-Эксперт", page_icon="🎓", layout="wide")

# Инициализация
try:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
except:
    st.error("Ошибка: Добавьте OPENROUTER_API_KEY в Secrets приложения.")
    st.stop()

def get_base64_image(image_file):
    image = Image.open(image_file)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Меню
page = st.sidebar.radio("Навигация", ["Главная", "Проверка"])

if page == "Главная":
    st.title("🎓 ЕГЭ-Эксперт")
    st.markdown("ИИ-анализ сочинений по критериям ФИПИ.")
else:
    st.title("✍️ Проверка сочинения")
    col1, col2 = st.columns([1, 1])
    with col1:
        source_text = st.text_area("Исходный текст:", height=100)
        input_type = st.radio("Источник:", ["Текст", "Фото"])
        essay_text = st.text_area("Текст сочинения:", height=200) if input_type == "Текст" else None
        uploaded_image = st.file_uploader("Загрузите фото:", type=["jpg", "png"]) if input_type == "Фото" else None

    with col2:
        if st.button("🚀 Проверить"):
            with st.spinner("Анализирую..."):
                # Упростили промпт, чтобы меньше ломался
                prompt = """
                Ты — старший эксперт ЕГЭ по русскому языку. Твоя задача: оценить сочинение строго по критериям ФИПИ, 
                как это делают на реальном экзамене.
                
                ПРИНЦИПЫ ПРОВЕРКИ:
                1. К1 (Проблема): Не придирайся к формулировке, если смысл передан верно.
                2. К2 (Комментарий): Ищи 2 примера-иллюстрации, связь между ними и их анализ. Если связь есть — это балл, даже если она не «гениальна».
                3. К3 (Позиция автора): Согласуется ли она с проблемой? Если да — балл.
                4. К5-К6 (Грамотность): Снимай баллы ТОЛЬКО за грубые фактические, логические и грамматические ошибки. 
                   Стилистические нюансы и «канцелярит» (если он не искажает смысл) — НЕ являются поводом для снижения баллов по критериям К5-К6.
                5. НЕ ИСПРАВЛЯЙ авторский стиль. Только выдели (жирным) действительно серьезные ошибки.
                
                Верни ответ ТОЛЬКО в формате JSON:
                {
                  "corrected_text": "Текст с выделенными **грубыми ошибками** (исправление)",
                  "table": {"К1": 0, "К2": 0, "К3": 0, "К4": 0, "К5": 0, "К6": 0, "К7": 0, "К8": 0, "К9": 0, "К10": 0, "К11": 0, "К12": 0},
                  "details": {"К1": "Обоснование по ФИПИ", ...},
                  "total": 0
                }
                """
                
                messages = [{"role": "user", "content": prompt + f" Исходный текст: {source_text}. Сочинение: {essay_text if essay_text else 'см. фото'}"}]
                if input_type == "Фото" and uploaded_image:
                    messages[0]["content"] = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(uploaded_image)}"}}]

                try:
                    response = client.chat.completions.create(model="google/gemma-4-31b-it:free", messages=messages)
                    content = response.choices[0].message.content
                    # Очистка от markdown
                    json_str = re.sub(r'^```json\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
                    data = json.loads(json_str)

                    # Вывод
                    st.metric("Итого", data.get("total", 0))
                    st.info(data.get("corrected_text", "Не удалось проанализировать текст."))
                    
                    st.subheader("Разбор")
                    table = data.get("table", {})
                    details = data.get("details", {})
                    for k in [f"К{i}" for i in range(1, 13)]:
                        with st.expander(f"{k} — {table.get(k, 0)} баллов"):
                            st.write(details.get(k, "Комментариев нет."))
                except Exception as e:
                    st.error(f"Ошибка: {e}. Попробуйте еще раз.")
