import streamlit as st
from openai import OpenAI
import json
import re
import base64
from PIL import Image
import io

# Настройка страницы
st.set_page_config(page_title="ЕГЭ 2026 Эксперт", page_icon="🎓", layout="wide")

# Инициализация клиента
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])

def get_base64_image(image_file):
    image = Image.open(image_file)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

st.title("✍️ Экспертиза сочинения ЕГЭ 2026")

col1, col2 = st.columns([1, 1])

with col1:
    source_text = st.text_area("Исходный текст:", height=100)
    input_type = st.radio("Источник:", ["Текст", "Фото"])
    essay_text = st.text_area("Сочинение:", height=200) if input_type == "Текст" else None
    uploaded_image = st.file_uploader("Загрузите фото:", type=["jpg", "png"]) if input_type == "Фото" else None

with col2:
    if st.button("🚀 Проверить по критериям 2026"):
        with st.spinner("Проверка по критериям ФИПИ 2026..."):
            # АКТУАЛЬНЫЙ ПРОМПТ 2026
            prompt = """
            Ты — эксперт ЕГЭ 2026. Оцени сочинение строго по актуальной шкале ФИПИ 2026.
            Критерии: К1 (Проблема), К2 (Комментарий), К3 (Отношение к позиции автора), К4 (Смысловая цельность), К5 (Точность и выразительность речи), К6 (Грамотность: орфография, пунктуация, грамматика, речевые нормы).
            
            Важно: соблюдай актуальное распределение баллов 2026 года.
            Верни ответ ТОЛЬКО в формате JSON:
            {
              "corrected_text": "Текст с правками **жирным**",
              "table": {"К1": 0, "К2": 0, "К3": 0, "К4": 0, "К5": 0, "К6": 0},
              "details": {"К1": "...", "К2": "...", "К3": "...", "К4": "...", "К5": "...", "К6": "..."},
              "total": 0
            }
            """
            
            messages = [{"role": "user", "content": prompt + f" Исходный текст: {source_text}. Сочинение: {essay_text if essay_text else 'см. фото'}"}]
            if input_type == "Фото" and uploaded_image:
                messages[0]["content"] = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(uploaded_image)}"}}]

            try:
                response = client.chat.completions.create(model="google/gemma-4-31b-it:free", messages=messages)
                content = response.choices[0].message.content
                json_str = re.sub(r'^```json\s*|\s*```$', '', content.strip(), flags=re.MULTILINE)
                data = json.loads(json_str)

                st.metric("Итого баллов (2026)", data.get("total", 0))
                st.info(data.get("corrected_text", "Текст не разобран"))
                
                st.subheader("📊 Разбор по критериям 2026")
                table = data.get("table", {})
                details = data.get("details", {})
                
                for k in ["К1", "К2", "К3", "К4", "К5", "К6"]:
                    with st.expander(f"{k} — {table.get(k, 0)} баллов"):
                        st.write(details.get(k, "Нет данных"))
            except Exception as e:
                st.error("Ошибка при проверке. Убедитесь, что текст четко виден на фото.")
