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
    st.error("Ошибка: API ключ не найден в Secrets.")
    st.stop()

# Функция обработки изображений
def get_base64_image(image_file):
    image = Image.open(image_file)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Меню навигации
page = st.sidebar.radio("Навигация", ["Главная", "Проверка сочинения"])

if page == "Главная":
    st.title("🎓 ЕГЭ-Эксперт: Система проверки сочинений")
    st.markdown("""
    Добро пожаловать в сервис для подготовки к ЕГЭ по русскому языку.
    Система проводит глубокий анализ вашего сочинения по **12 критериям ФИПИ**.
    
    **Что вы получите:**
    1. Текст с выделенными ошибками и комментариями к ним.
    2. Таблицу с баллами по каждому критерию.
    3. Детальные советы, как улучшить работу.
    """)
    st.info("Вы можете загрузить как готовый текст, так и фотографию вашей рукописи.")

else:
    st.title("✍️ Проверка сочинения")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        source_text = st.text_area("Исходный текст (для анализа):", height=100)
        input_type = st.radio("Как загрузить сочинение?", ["Текст", "Фото рукописи"])
        
        essay_text = None
        uploaded_image = None
        
        if input_type == "Текст":
            essay_text = st.text_area("Текст сочинения:", height=200)
        else:
            uploaded_image = st.file_uploader("Загрузите фото страницы...", type=["jpg", "jpeg", "png"])

    with col2:
        if st.button("🚀 Проверить сочинение"):
            if not source_text or (input_type == "Текст" and not essay_text) or (input_type == "Фото рукописи" and not uploaded_image):
                st.warning("Пожалуйста, заполните все поля!")
            else:
                with st.spinner("Анализирую текст и фото..."):
                    prompt = f"""
                    Ты — строгий эксперт ЕГЭ. Исходный текст: {source_text}. 
                    Проанализируй сочинение. Верни JSON-ответ с полями:
                    - 'corrected_text': текст, где ошибки выделены жирным **...** (рядом в скобках исправление).
                    - 'table': объект с полями 'Критерий' (К1...К12) и 'Баллы' (массив чисел).
                    - 'details': объект, где ключ — критерий (К1..К12), значение — текст с разбором (цитата ошибки, почему это ошибка, как исправить).
                    - 'total': итоговый балл.
                    
                    Отвечай ТОЛЬКО в формате JSON:
                    """
                    
                    image_b64 = None
                    if input_type == "Фото рукописи":
                        image_b64 = get_base64_image(uploaded_image)
                        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]}]
                    else:
                        messages = [{"role": "user", "content": f"{prompt} Сочинение: {essay_text}"}]

                    response = client.chat.completions.create(model="google/gemma-4-31b-it:free", messages=messages)
                    data = json.loads(response.choices[0].message.content.replace('```json', '').replace('```', ''))

                    # ВЫВОД РЕЗУЛЬТАТОВ
                    st.success("Проверка завершена!")
                    st.metric("Итого баллов", data["total"])
                    
                    st.subheader("📝 Анализ текста с правками")
                    st.info(data["corrected_text"])
                    
                    st.subheader("📊 Разбор по критериям")
                    for k in data["details"]:
                        bal = next(item["Баллы"] for item in data["table"] if item["Критерий"] == k) # упрощено
                        with st.expander(f"{k} — {bal} балл(ов)"):
                            st.write(data["details"][k])
