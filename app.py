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

# Навигация
page = st.sidebar.radio("Навигация", ["Главная", "Проверка"])

if page == "Главная":
    st.title("🎓 ЕГЭ-Эксперт")
    st.markdown("""
    Сервис для проверки сочинений по стандартам ФИПИ.
    - **Объективность:** проверка логики, структуры и аргументации.
    - **Экспертность:** фокус на критериях, а не на мелких стилистических правках.
    """)
else:
    st.title("✍️ Проверка сочинения")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        source_text = st.text_area("Исходный текст (обязательно):", height=100)
        input_type = st.radio("Источник:", ["Текст", "Фото"])
        essay_text = st.text_area("Ваше сочинение:", height=200) if input_type == "Текст" else None
        uploaded_image = st.file_uploader("Загрузите фото рукописи:", type=["jpg", "png"]) if input_type == "Фото" else None

    with col2:
        if st.button("🚀 Проверить по критериям ФИПИ"):
            with st.spinner("Работает эксперт..."):
                # ЭКСПЕРТНЫЙ ПРОМПТ
                prompt = """
                Ты — старший эксперт ЕГЭ по русскому языку. Оцени сочинение строго по методике ФИПИ.
                ПРИНЦИПЫ:
                1. К2: Проверяй наличие 2 примеров-иллюстраций, их анализа и смысловой связи.
                2. К5-К6: Снимай баллы ТОЛЬКО за грубые логические, фактические и грамматические ошибки. Стилистические шероховатости — НЕ повод для снижения.
                3. Не переписывай стиль автора. Исправляй только то, что ведет к потере баллов.
                
                Верни ответ ТОЛЬКО в формате JSON:
                {
                  "corrected_text": "Текст, где **грубые ошибки** исправлены в скобках",
                  "table": {"К1": 0, "К2": 0, "К3": 0, "К4": 0, "К5": 0, "К6": 0, "К7": 0, "К8": 0, "К9": 0, "К10": 0, "К11": 0, "К12": 0},
                  "details": {"К1": "Обоснование оценки", "К2": "...", ...},
                  "total": 0
                }
                """
                
                messages = [{"role": "user", "content": prompt + f" Исходный текст: {source_text}. Сочинение: {essay_text if essay_text else 'см. фото'}"}]
                if input_type == "Фото" and uploaded_image:
                    messages[0]["content"] = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(uploaded_image)}"}}]

                try:
                    response = client.chat.completions.create(model="google/gemma-4-31b-it:free", messages=messages)
                    content = response.choices[0].message.content
                    # Очистка JSON
                    json_str = re.sub(r'^```json\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
                    data = json.loads(json_str)

                    # Вывод
                    st.metric("Итого баллов", data.get("total", 0))
                    with st.expander("📝 Анализ текста с правками", expanded=True):
                        st.info(data.get("corrected_text", "Ошибка разбора текста."))
                    
                    st.subheader("📊 Разбор по критериям ФИПИ")
                    table = data.get("table", {})
                    details = data.get("details", {})
                    
                    for k in [f"К{i}" for i in range(1, 13)]:
                        bal = table.get(k, 0)
                        status = "✅" if bal > 0 else "❌"
                        with st.expander(f"{k} — {bal} баллов {status}"):
                            st.write(details.get(k, "Комментариев нет."))
                except Exception as e:
                    st.error(f"Ошибка: ИИ вернул ответ в непредвиденном формате. Попробуйте еще раз.")
