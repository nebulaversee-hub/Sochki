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
# Стилизация: образ школьной тетради и экзаменационного бланка.
# Шрифты PT Serif / PT Sans / PT Mono — шрифты ParaType с поддержкой
# кириллицы, исторически используемые в российских учебных изданиях.
# Красная линия слева — отсылка к полям школьной тетради; зелёный —
# к обложке "ученической тетради" и официальным бланкам.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=PT+Serif:ital,wght@0,400;0,700;1,400&family=PT+Sans:wght@400;700&family=PT+Mono&display=swap');

    :root {
        --paper: #FAF7EF;
        --paper-light: #FFFEFA;
        --ink: #20262B;
        --green: #2E5339;
        --green-dark: #1D3A26;
        --red-pen: #B23A2E;
        --gold: #C9A227;
        --rule: #DCD5C4;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--paper) !important;
    }
    [data-testid="stAppViewContainer"] .block-container {
        padding-top: 1.2rem;
        max-width: 1100px;
    }
    body, p, span, label, div {
        font-family: 'PT Sans', sans-serif;
        color: var(--ink);
    }

    /* Заголовок-«обложка тетради» */
    .exam-banner {
        background: var(--green);
        color: var(--paper);
        padding: 1.1rem 1.6rem;
        border-radius: 4px;
        margin-bottom: 1.6rem;
        border-left: 6px solid var(--red-pen);
    }
    .exam-banner h1 {
        font-family: 'PT Serif', serif;
        font-weight: 700;
        font-size: 1.7rem;
        margin: 0;
        color: var(--paper);
        letter-spacing: 0.01em;
    }
    .exam-banner p {
        margin: 0.3rem 0 0 0;
        font-size: 0.92rem;
        color: var(--paper);
        opacity: 0.85;
    }

    /* Поля ввода текста — «линованная бумага» с красными полями слева */
    [data-testid="stTextArea"] textarea {
        background-color: var(--paper-light);
        background-image: repeating-linear-gradient(
            180deg, transparent, transparent 27px, var(--rule) 28px
        );
        line-height: 28px;
        border: 1px solid var(--rule);
        border-left: 4px solid var(--red-pen);
        border-radius: 2px;
        font-family: 'PT Sans', sans-serif;
        color: var(--ink);
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--green);
        box-shadow: 0 0 0 1px var(--green);
    }

    /* Кнопки — официально-зелёные */
    .stButton button, .stDownloadButton button {
        background-color: var(--green);
        color: var(--paper) !important;
        border: none;
        border-radius: 3px;
        font-family: 'PT Sans', sans-serif;
        font-weight: 700;
        letter-spacing: 0.02em;
        padding: 0.55rem 1.2rem;
        transition: background-color 0.15s ease;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background-color: var(--green-dark);
        color: var(--paper) !important;
    }

    /* Карточка с проверенным сочинением */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--paper-light);
        border: 1px solid var(--rule) !important;
        border-left: 5px solid var(--red-pen) !important;
        border-radius: 3px;
        padding: 0.5rem 0.8rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] p {
        font-family: 'PT Serif', serif;
        font-size: 1.05rem;
        line-height: 1.75;
        color: var(--ink);
    }
    [data-testid="stVerticalBlockBorderWrapper"] strong {
        color: var(--red-pen);
        text-decoration: underline wavy var(--red-pen);
        text-underline-offset: 3px;
    }

    /* Баллы по критериям — клетки экзаменационного бланка */
    [data-testid="stNumberInput"] label p {
        font-family: 'PT Mono', monospace;
        font-weight: 700;
        color: var(--green-dark);
        font-size: 0.8rem;
    }
    [data-testid="stNumberInput"] input {
        font-family: 'PT Mono', monospace;
        border: 1px solid var(--rule);
        border-radius: 2px;
        background-color: var(--paper-light);
    }
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--green);
        box-shadow: 0 0 0 1px var(--green);
    }

    /* Итоговый балл — «печать» */
    [data-testid="stMetric"] {
        background: var(--paper-light);
        border: 2px solid var(--gold);
        border-radius: 10px;
        padding: 0.8rem 1.4rem;
        width: fit-content;
    }
    [data-testid="stMetricValue"] {
        font-family: 'PT Serif', serif;
        color: var(--green-dark) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'PT Mono', monospace;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 0.75rem;
    }

    /* Уведомления — приглушённый «красный карандаш» вместо ярко-розового */
    div[data-baseweb="notification"] {
        border-left: 4px solid var(--red-pen) !important;
        border-radius: 2px;
        font-family: 'PT Sans', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# Промпт для проверки сочинения. При желании отредактируйте формулировки
# критериев под свою методику — структура JSON в конце менять не нужно,
# она используется кодом для разбора ответа.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Оцени сочинение ученика по критериям ЕГЭ К1–К12.

ИСХОДНЫЙ ТЕКСТ:
{source_text}

СОЧИНЕНИЕ УЧЕНИКА:
{essay_text}

Верни ТОЛЬКО JSON-объект такой структуры (никакого другого текста):
{{"corrected_text":"текст сочинения с исправлениями, ошибки выделены **жирным**","scores":{{"K1":0,"K2":0,"K3":0,"K4":0,"K5":0,"K6":0,"K7":0,"K8":0,"K9":0,"K10":0,"K11":0,"K12":0}},"comments":{{"K1":"","K2":"","K3":"","K4":"","K5":"","K6":"","K7":"","K8":"","K9":"","K10":"","K11":"","K12":""}}}}
"""


def call_claude_api(source_text: str, essay_text: str, model_name: str) -> str:
    """Отправляет запрос через OpenRouter и возвращает текст ответа."""
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
    prompt = PROMPT_TEMPLATE.format(source_text=source_text, essay_text=essay_text)
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=4000,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a JSON-only response bot. "
                    "You MUST respond with a single valid JSON object and nothing else. "
                    "No explanations, no markdown, no code blocks, no text before or after the JSON. "
                    "Your response must start with { and end with }."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def extract_json(raw_text: str):
    """
    Безопасно извлекает JSON-объект из ответа модели, даже если он:
    - обёрнут в markdown-блок ```json ... ```
    - содержит пояснительный текст до или после JSON
    - содержит фигурные скобки внутри пояснительного текста
    Пробует декодировать JSON начиная с каждой найденной "{", пока не
    получится валидный объект. Возвращает dict или None, если не вышло.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            break
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        search_from = start + 1

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
st.markdown(
    """
    <div class="exam-banner">
        <h1>📝 Проверка сочинений ЕГЭ</h1>
        <p>Разбор по критериям К1–К12 с помощью ИИ</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    with st.container(border=True):
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
