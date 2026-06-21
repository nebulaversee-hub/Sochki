"""
Приложение для проверки сочинений ЕГЭ с помощью Claude (через OpenRouter).

Запуск:
    streamlit run ege_essay_checker.py

Настройка:
    В .streamlit/secrets.toml укажите:
        OPENROUTER_API_KEY = "sk-or-..."
        OPENROUTER_MODEL = "google/gemma-4-31b-it:free"   # необязательно, есть значение по умолчанию
"""

import json
import re
from datetime import datetime

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Проверка сочинений ЕГЭ", layout="wide")

# ---------------------------------------------------------------------------
# Структура критериев (K1–K12). Здесь хранится ТОЛЬКО рубрика ФИПИ
# (название критерия и верхняя граница шкалы) — сами баллы за конкретное
# сочинение нигде не зашиты и всегда вводятся пользователем / предлагаются ИИ.
# ---------------------------------------------------------------------------
CRITERIA = {
    "K1": "Соответствие теме",
    "K2": "Комментарий к теме",
    "K3": "Отражение позиции автора исходного текста",
    "K4": "Отношение к позиции автора, аргументация своего мнения",
    "K5": "Смысловая цельность, речевая связность и последовательность изложения",
    "K6": "Точность и выразительность речи",
    "K7": "Орфографические нормы",
    "K8": "Пунктуационные нормы",
    "K9": "Языковые нормы",
    "K10": "Речевые нормы",
    "K11": "Этические нормы",
    "K12": "Фактологическая точность",
}

MAX_SCORE = {
    "K1": 1, "K2": 3, "K3": 1, "K4": 1, "K5": 2, "K6": 2,
    "K7": 3, "K8": 3, "K9": 2, "K10": 2, "K11": 1, "K12": 1,
}

# ---------------------------------------------------------------------------
# !!! ВСТАВЬТЕ СВОЙ ПРОМПТ СЮДА !!!
# Промпт обязательно должен требовать от модели ответ строго в виде JSON
# следующей структуры (без лишнего текста до/после):
#
# {
#   "corrected_text": "текст сочинения, где исправленные места выделены **жирным** в markdown",
#   "scores": {"K1": 1, "K2": 2, ..., "K12": 1},
#   "comments": {"K1": "краткий комментарий", ..., "K12": "краткий комментарий"}
# }
#
# Плейсхолдеры {source_text} и {essay_text} подставляются автоматически.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """ВАШ ПРОМПТ ЗДЕСЬ.

ИСХОДНЫЙ ТЕКСТ:
{source_text}

СОЧИНЕНИЕ УЧЕНИКА:
{essay_text}
"""


def call_claude_api(source_text: str, essay_text: str, model_name: str) -> str:
    """Отправляет запрос в Claude (через OpenRouter) и возвращает текст ответа."""
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
    prompt = PROMPT_TEMPLATE.format(source_text=source_text, essay_text=essay_text)
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def extract_json(raw_text: str):
    """
    Безопасно извлекает JSON-объект из ответа модели, даже если он:
    - обёрнут в markdown-блок ```json ... ```
    - содержит пояснительный текст до или после JSON
    Возвращает dict или None, если разобрать не удалось.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Состояние сессии
# ---------------------------------------------------------------------------
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "raw_response" not in st.session_state:
    st.session_state.raw_response = ""

# ---------------------------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------------------------
st.title("📝 Проверка сочинений ЕГЭ с помощью ИИ")

with st.sidebar:
    st.header("Настройки")
    model_name = st.text_input(
        "Модель (слаг OpenRouter)",
        value=st.secrets.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
        help="Проверьте актуальный слаг модели на openrouter.ai/models",
    )
    st.caption("API-ключ берётся из st.secrets и не отображается в интерфейсе.")

col1, col2 = st.columns(2)
with col1:
    source_text = st.text_area("Исходный текст", height=320, key="source_text")
with col2:
    essay_text = st.text_area("Текст сочинения", height=320, key="essay_text")

check_clicked = st.button("Проверить", type="primary", use_container_width=True)

if check_clicked:
    if not source_text.strip() or not essay_text.strip():
        st.warning("Заполните оба поля — исходный текст и сочинение.")
    elif "OPENROUTER_API_KEY" not in st.secrets:
        st.error("Не найден API-ключ. Добавьте OPENROUTER_API_KEY в st.secrets.")
    else:
        raw = None
        with st.spinner("Анализирую сочинение..."):
            try:
                raw = call_claude_api(source_text, essay_text, model_name)
                st.session_state.raw_response = raw
            except Exception as e:
                st.error(f"Ошибка при обращении к API: {e}")
                st.session_state.ai_result = None

        if raw:
            parsed = extract_json(raw)
            if parsed is None:
                st.error("Не удалось разобрать JSON-ответ от модели.")
                with st.expander("Показать необработанный ответ"):
                    st.code(raw)
                st.session_state.ai_result = None
            else:
                st.session_state.ai_result = parsed

# ---------------------------------------------------------------------------
# Отображение результата
# ---------------------------------------------------------------------------
result = st.session_state.ai_result

if result:
    st.divider()
    st.subheader("Сочинение с исправлениями")
    corrected = result.get("corrected_text", "Модель не вернула исправленный текст.")
    st.markdown(corrected)

    ai_scores = result.get("scores", {}) or {}
    ai_comments = result.get("comments", {}) or {}

    st.divider()
    st.subheader("Баллы по критериям (K1–K12)")
    st.caption(
        "Баллы, предложенные ИИ, подставлены как отправная точка — "
        "вы можете свободно изменить любое значение."
    )

    score_cols = st.columns(3)
    user_scores = {}
    for i, k in enumerate(CRITERIA.keys()):
        with score_cols[i % 3]:
            try:
                default_value = int(ai_scores.get(k, 0))
            except (TypeError, ValueError):
                default_value = 0
            default_value = max(0, min(default_value, MAX_SCORE[k]))

            user_scores[k] = st.number_input(
                f"{k}. {CRITERIA[k]} (макс. {MAX_SCORE[k]})",
                min_value=0,
                max_value=MAX_SCORE[k],
                value=default_value,
                step=1,
                key=f"score_{k}",
            )
            comment = ai_comments.get(k)
            if comment:
                st.caption(comment)

    total_score = sum(user_scores.values())
    max_total = sum(MAX_SCORE.values())

    st.divider()
    st.metric("Итоговый балл", f"{total_score} / {max_total}")

    report_lines = [
        "ОТЧЁТ ПО ПРОВЕРКЕ СОЧИНЕНИЯ ЕГЭ",
        f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        "",
        "=== ИСХОДНЫЙ ТЕКСТ ===",
        source_text,
        "",
        "=== СОЧИНЕНИЕ С ИСПРАВЛЕНИЯМИ ===",
        corrected,
        "",
        "=== БАЛЛЫ ПО КРИТЕРИЯМ ===",
    ]
    for k in CRITERIA:
        line = f"{k} ({CRITERIA[k]}): {user_scores[k]} из {MAX_SCORE[k]}"
        if ai_comments.get(k):
            line += f" — {ai_comments[k]}"
        report_lines.append(line)
    report_lines.append("")
    report_lines.append(f"ИТОГО: {total_score} из {max_total}")

    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Скачать отчёт",
        data=report_text,
        file_name=f"otchet_sochinenie_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )
elif st.session_state.raw_response and not check_clicked:
    with st.expander("Необработанный ответ модели (для отладки)"):
        st.code(st.session_state.raw_response)
