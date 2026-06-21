import streamlit as st
from openai import OpenAI
import json
import pandas as pd

# Получаем ключ из настроек Streamlit (Secrets)
# В Streamlit Cloud это настраивается в разделе: Manage App -> Settings -> Secrets
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except:
    st.error("Ошибка: API ключ не найден. Добавьте OPENROUTER_API_KEY в Secrets.")
    st.stop()

# Инициализация клиента
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=api_key
)

st.set_page_config(page_title="ЕГЭ-Эксперт", page_icon="🎓")
st.title("🎓 ЕГЭ-Эксперт: Проверка сочинений")

# Поля для ввода
source_text = st.text_area("Вставьте исходный текст для анализа:", height=150)
essay_text = st.text_area("Вставьте сочинение ученика:", height=200)

if st.button("Проверить сочинение"):
    if not source_text or not essay_text:
        st.warning("Пожалуйста, заполните оба поля.")
    else:
        with st.spinner("Анализирую по критериям ФИПИ..."):
            prompt = f"""
            Ты — строгий эксперт ЕГЭ по русскому языку. Оцени сочинение строго по критериям ФИПИ (К1-К12).
            Исходный текст: {source_text}
            Сочинение: {essay_text}
            
            Верни ответ строго в формате JSON, без пояснительного текста:
            {{
              "table": {{"Критерий": ["К1", "К2", "К3", "К4", "К5", "К6", "К7", "К8", "К9", "К10", "К11", "К12"], "Баллы": [0,0,0,0,0,0,0,0,0,0,0,0]}},
              "total": 0,
              "comment": "Подробный разбор ошибок и рекомендации"
            }}
            """
            
            try:
                response = client.chat.completions.create(
                    model="google/gemma-4-31b-it:free", # Твоя модель
                    messages=[{"role": "user", "content": prompt}],
                )
                
                # Обработка ответа
                content = response.choices[0].message.content.replace('```json', '').replace('```', '')
                data = json.loads(content)
                
                # Отображение данных
                st.subheader("📊 Результаты оценки")
                df = pd.DataFrame(data["table"])
                st.table(df)
                
                st.metric("Итого баллов", data["total"])
                
                st.subheader("📝 Разбор эксперта")
                st.write(data["comment"])
                
            except Exception as e:
                st.error(f"Произошла ошибка при анализе: {e}")
