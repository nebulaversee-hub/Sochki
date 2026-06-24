"""
ЕГЭ Эксперт — проверка сочинений по русскому языку (задание 27).

Запуск:     streamlit run ege_essay_checker.py
Зависимости: pip install streamlit openai

.streamlit/secrets.toml:
    OPENROUTER_API_KEY    = "sk-or-..."
    OPENROUTER_MODEL      = "google/gemma-4-31b-it:free"        # текстовая проверка
    OPENROUTER_VISION_MODEL = "google/gemma-4-31b-it:free"  # распознавание рукописи
"""

import base64
import json
import re
from datetime import datetime

import streamlit as st
from openai import OpenAI

# ══════════════════════════════════════════════════════════
#  Конфигурация
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ЕГЭ Эксперт",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════
#  Состояние сессии
# ══════════════════════════════════════════════════════════
_defaults = {
    "page": "home",
    "result": None,
    "raw": "",
    "input_mode": "text",          # "text" | "photo"
    "transcription": "",           # расшифровка рукописи от ИИ
    "transcription_done": False,   # шаг 1 выполнен
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════
#  Критерии ФИПИ 2026
# ══════════════════════════════════════════════════════════
CRITERIA = {
    "K1":  ("Отражение позиции автора", 1),
    "K2":  ("Комментарий к позиции автора", 3),
    "K3":  ("Собственное отношение и обоснование", 2),
    "K4":  ("Фактическая точность", 1),
    "K5":  ("Логичность речи", 2),
    "K6":  ("Этические нормы", 1),
    "K7":  ("Орфографические нормы", 3),
    "K8":  ("Пунктуационные нормы", 3),
    "K9":  ("Грамматические нормы", 3),
    "K10": ("Речевые нормы", 3),
}
MAX_TOTAL = sum(v[1] for v in CRITERIA.values())  # 22

# ══════════════════════════════════════════════════════════
#  Промпты
# ══════════════════════════════════════════════════════════

# Промпт для распознавания рукописи (возвращает чистый текст, не JSON)
TRANSCRIBE_PROMPT = """
На изображении — рукописное сочинение ученика ЕГЭ по русскому языку.

Расшифруй текст максимально точно.

Правила:
• Сохраняй абзацы.
• Сохраняй все знаки препинания.
• Не исправляй ошибки ученика.
• Если слово неразборчиво — используй [???].
• Если знак препинания неразборчив — используй [???].
• Не придумывай текст.
• Верни только текст сочинения.
"""

# Промпт для проверки сочинения (возвращает JSON)
PROMPT_TEMPLATE = """\
Ты — эксперт ЕГЭ по русскому языку. Проверь сочинение-рассуждение (задание 27) по критериям ФИПИ 2026.

КРИТЕРИИ (К1–К10, максимум 22 балла):
К1 (макс 1): Отражение позиции автора. 1 — верно; 0 — нет. Если К1=0, то К2=К3=0.
К2 (макс 3): Комментарий. 3 — 2 примера+пояснение+связь с пояснением; 2 — 2 примера+пояснение, связь без пояснения; 1 — 1 пример с пояснением; 0 — иначе.
К3 (макс 2): Своё отношение. 2 — обосновано+пример-аргумент; 1 — обосновано без примера; 0 — только формальное.
К4 (макс 1): Факты. 1 — ошибок нет; 0 — есть.
К5 (макс 2): Логика. 2 — ошибок нет; 1 — 1–2; 0 — 3 и более.
К6 (макс 1): Этика. 1 — нарушений нет; 0 — есть.
К7 (макс 3): Орфография. 3—0 ош.; 2—1–2; 1—3–4; 0—5+.
К8 (макс 3): Пунктуация. 3—0 ош.; 2—1–2; 1—3–4; 0—5+.
К9 (макс 3): Грамматика. 3—0 ош.; 2—1–2; 1—3–4; 0—5+.
К10 (макс 3): Речь. 3—0 ош.; 2—1–2; 1—3–4; 0—5+.

ИСХОДНЫЙ ТЕКСТ:
{source_text}

СОЧИНЕНИЕ УЧЕНИКА:
{essay_text}

Верни ТОЛЬКО JSON (без ```, без пояснений вне JSON):
{{"corrected_text":"полный текст сочинения, каждая ошибка выделена **жирным**","scores":{{"K1":0,"K2":0,"K3":0,"K4":0,"K5":0,"K6":0,"K7":0,"K8":0,"K9":0,"K10":0}},"comments":{{"K1":"","K2":"","K3":"","K4":"","K5":"","K6":"","K7":"","K8":"","K9":"","K10":""}}}}
"""

# ══════════════════════════════════════════════════════════
#  API
# ══════════════════════════════════════════════════════════
def _client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )


def transcribe_image(img_bytes: bytes, img_mime: str) -> str:
    """Шаг 1: отправляет фото рукописи в vision-модель, возвращает расшифрованный текст."""
    client = _client()
    model = st.secrets.get("OPENROUTER_VISION_MODEL", "google/gemma-4-31b-it:free")
    b64 = base64.b64encode(img_bytes).decode()
    data_url = f"data:{img_mime};base64,{b64}"
    r = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise handwriting transcription assistant. "
                    "Return ONLY the transcribed text, nothing else — no commentary, no JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                ],
            },
        ],
    )
    return r.choices[0].message.content.strip()


def call_api(source_text: str, essay_text: str) -> str:
    """Шаг 2 (и единственный для текстового режима): проверка сочинения, возвращает JSON."""
    client = _client()
    model = st.secrets.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    prompt = PROMPT_TEMPLATE.format(source_text=source_text, essay_text=essay_text)
    r = client.chat.completions.create(
        model=model,
        max_tokens=4000,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a JSON-only bot. "
                    "Respond ONLY with a valid JSON object. "
                    "No explanations, no markdown, no code blocks. "
                    "Response must start with { and end with }."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return r.choices[0].message.content


def extract_json(raw: str):
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    dec = json.JSONDecoder()
    pos = 0
    while True:
        s = text.find("{", pos)
        if s == -1:
            break
        try:
            obj, _ = dec.raw_decode(text, s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        pos = s + 1
    return None


def md_bold_to_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text, flags=re.DOTALL)

# ══════════════════════════════════════════════════════════
#  Сборщики HTML
# ══════════════════════════════════════════════════════════
def build_score_cards(result: dict) -> str:
    scores   = result.get("scores", {}) or {}
    comments = result.get("comments", {}) or {}
    html = '<div class="sc-grid">'
    for i, (k, (name, max_s)) in enumerate(CRITERIA.items()):
        try:
            score = max(0, min(int(scores.get(k, 0)), max_s))
        except (TypeError, ValueError):
            score = 0
        comment = str(comments.get(k, "")).strip()
        cls = "sc-green" if score == max_s else ("sc-amber" if score > 0 else "sc-red")
        dots = "".join(
            f'<span class="dot{" filled" if j < score else ""}"></span>'
            for j in range(max_s)
        )
        delay = i * 0.05
        html += f"""
<div class="sc-card {cls}" style="animation-delay:{delay:.2f}s">
  <div class="sc-head">
    <span class="sc-k">{k}</span>
    <span class="sc-frac">{score}/{max_s}</span>
  </div>
  <div class="sc-name">{name}</div>
  <div class="sc-dots">{dots}</div>
  {"" if not comment else f'<div class="sc-comment">{comment}</div>'}
</div>"""
    html += "</div>"
    return html


def build_total(total: int) -> str:
    pct = total / MAX_TOTAL * 100
    if pct >= 82:
        v, s = "Отличный результат", "Сочинение соответствует высокому уровню ЕГЭ"
    elif pct >= 60:
        v, s = "Хороший результат", "Есть небольшие недочёты, которые можно улучшить"
    elif pct >= 40:
        v, s = "Требует доработки", "Обратите внимание на комментарии к критериям"
    else:
        v, s = "Необходима работа над ошибками", "Рекомендуем детально проработать каждый критерий"
    return f"""
<div class="total-wrap">
  <div class="total-stamp">
    <div class="total-num">{total}</div>
    <div class="total-den">из {MAX_TOTAL}</div>
  </div>
  <div class="total-text">
    <h3>{v}</h3>
    <p>{s}</p>
  </div>
</div>"""


def build_report(src: str, essay: str, result: dict, total: int) -> str:
    scores   = result.get("scores", {}) or {}
    comments = result.get("comments", {}) or {}
    lines = [
        "══════════════════════════════════════════",
        "   ОТЧЁТ ПО ПРОВЕРКЕ СОЧИНЕНИЯ ЕГЭ 2026",
        f"   Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        "══════════════════════════════════════════",
        "", "ИСХОДНЫЙ ТЕКСТ:", src, "",
        "СОЧИНЕНИЕ:", essay, "",
        "── БАЛЛЫ ПО КРИТЕРИЯМ ФИПИ 2026 ──", "",
    ]
    for k, (name, max_s) in CRITERIA.items():
        sc = scores.get(k, 0)
        cm = comments.get(k, "")
        lines.append(f"{k}. {name}: {sc}/{max_s}")
        if cm:
            lines.append(f"   {cm}")
    lines += ["", f"ИТОГО: {total} / {MAX_TOTAL}",
              "══════════════════════════════════════════"]
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════
#  CSS — тёмная монотонная тема
# ══════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Сброс Streamlit ── */
#MainMenu, footer, header[data-testid="stHeader"] { display:none !important; }
[data-testid="collapsedControl"] { display:none !important; }
.main .block-container { padding:0 !important; max-width:100% !important; }
*, *::before, *::after { box-sizing:border-box; }

/* ── Палитра ── */
:root {
  --bg:       #FFFFFF;
  --bg2:      #FFFFFF;
  --card:     #FFFFFF;
  --border:   #D9D9D9;
  --text:     #111111;
  --muted:    #555555;
  --accent:   #DC2626;
  --accent-d: rgba(220,38,38,.08);
}

html, body { font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); margin:0; }

/* ── Анимации ── */
@keyframes fadeUp   { from{opacity:0;transform:translateY(26px)} to{opacity:1;transform:translateY(0)} }
@keyframes drawLine { from{width:0} to{width:100%} }
@keyframes stampIn  {
  0%  {opacity:0;transform:rotate(-14deg) scale(2.4)}
  70% {opacity:1;transform:rotate(3deg) scale(0.95)}
  100%{opacity:1;transform:rotate(-8deg) scale(1)}
}
@keyframes slideCard{from{opacity:0;transform:translateX(-14px)} to{opacity:1;transform:translateX(0)}}
@keyframes fadeIn   {from{opacity:0}to{opacity:1}}
@keyframes pulse    {
  0%,100%{box-shadow:0 0 0 0 rgba(220,38,38,.4)}
  50%    {box-shadow:0 0 0 10px rgba(220,38,38,0)}
}

/* ══════════ ГЛАВНАЯ ══════════ */

.hero {
  background: var(--bg);
  padding: 88px 6% 96px;
  position: relative; overflow: hidden;
}
.hero::before {
  content:''; position:absolute; inset:0;
  background:
    radial-gradient(ellipse 50% 50% at 10% 60%, rgba(220,38,38,.07) 0%, transparent 65%),
    radial-gradient(ellipse 35% 35% at 92% 30%, rgba(99,102,241,.05) 0%, transparent 65%);
}
.hero::after {
  content:''; position:absolute; inset:0;
  background-image:
    linear-gradient(rgba(255,255,255,.012) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.012) 1px, transparent 1px);
  background-size:46px 46px;
}
.hero-inner {
  position:relative; z-index:2; max-width:1160px; width:100%;
  display:grid; grid-template-columns:1fr 380px;
  gap:72px; align-items:center; margin:0 auto;
}
.eyebrow {
  display:inline-flex; align-items:center; gap:8px;
  background:var(--accent-d); border:1px solid rgba(220,38,38,.28);
  color:#FCA5A5; font-size:.71rem; font-weight:700;
  letter-spacing:.13em; text-transform:uppercase;
  padding:6px 14px; border-radius:100px; margin-bottom:28px;
  animation:fadeUp .6s ease both;
}
.hero-h1 {
  font-family:'Playfair Display',serif;
  font-size:clamp(2.7rem,4.3vw,4.2rem); font-weight:900;
  color:#111111; line-height:1.06; margin:0;
  animation:fadeUp .65s .08s ease both;
}
.accent {
  color:var(--accent); position:relative; display:inline-block;
}
.accent::after {
  content:''; position:absolute; bottom:2px; left:0;
  height:3px; background:var(--accent); border-radius:2px;
  animation:drawLine .9s 1s ease both; width:0;
}
.hero-sub {
  color:var(--muted); font-size:1.03rem; line-height:1.74;
  margin:22px 0 0; max-width:450px;
  animation:fadeUp .65s .15s ease both;
}
.hero-badges {
  display:flex; gap:22px; margin-top:28px;
  animation:fadeUp .65s .23s ease both;
}
.hbadge       { display:flex; align-items:center; gap:7px; color:rgba(255,255,255,.32); font-size:.79rem; }
.hbadge-dot   { width:5px; height:5px; background:var(--accent); border-radius:50%; flex-shrink:0; }

/* Макет экзаменационного бланка */
.paper {
  background:#FFFEFA; border-radius:6px;
  padding:26px 22px 76px;
  box-shadow:0 28px 80px rgba(0,0,0,.7), 0 0 0 1px rgba(255,255,255,.03);
  animation:fadeUp .7s .28s ease both;
  transform:rotate(1.8deg); position:relative;
}
.paper::before {
  content:'ЕГЭ 2026 · Задание 27';
  position:absolute; top:-10px; left:18px;
  background:var(--accent); color:#111111;
  font-size:.6rem; font-weight:700; letter-spacing:.08em;
  padding:3px 10px; border-radius:3px;
}
.paper-h { font-family:'Playfair Display',serif; font-size:.8rem; font-weight:700; color:#1E3A5F; margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid #E5E0D8; }
.paper-t { font-size:.75rem; color:#374151; line-height:2; }
.paper-t .e { color:var(--accent); text-decoration:underline wavy var(--accent); text-underline-offset:3px; font-weight:600; }
.pscores { margin-top:14px; display:grid; grid-template-columns:repeat(5,1fr); gap:5px; }
.ps      { background:#F3F4F6; border:1px solid #E5E7EB; border-radius:4px; text-align:center; padding:5px 0; }
.ps .pk  { font-size:.5rem; color:#9CA3AF; font-weight:700; letter-spacing:.05em; display:block; }
.ps .pv  { font-family:'JetBrains Mono',monospace; font-size:.82rem; font-weight:700; display:block; color:#111827; }
.ps.g   { background:#ECFDF5; border-color:#6EE7B7; } .ps.g .pv{color:#059669;}
.ps.a   { background:#FFFBEB; border-color:#FCD34D; } .ps.a .pv{color:#D97706;}
.ps.r   { background:#FEF2F2; border-color:#FCA5A5; } .ps.r .pv{color:var(--accent);}
.paper-stamp {
  position:absolute; bottom:18px; right:16px;
  width:66px; height:66px; border-radius:50%;
  border:3px solid var(--accent);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  color:var(--accent); animation:stampIn .5s 1.35s ease both;
  opacity:0; animation-fill-mode:both;
}
.ps-n { font-family:'JetBrains Mono',monospace; font-size:1.25rem; font-weight:700; line-height:1; }
.ps-d { font-size:.48rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; opacity:.75; }

/* Общие элементы секций */
.sec        { max-width:1160px; margin:0 auto; }
.sec-label  { font-size:.7rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; color:var(--accent); margin-bottom:10px; }
.sec-title  { font-family:'Playfair Display',serif; font-size:clamp(1.7rem,2.8vw,2.4rem); font-weight:700; color:#111111; margin:0 0 14px; max-width:480px; }
.sec-sub    { color:var(--muted); font-size:.95rem; line-height:1.7; max-width:440px; margin-bottom:50px; }
.divider    { border:none; border-top:1px solid var(--border); margin:0; }

/* Секция возможностей */
.features   { background:var(--bg); padding:90px 6%; border-top:1px solid var(--border); }
.feat-grid  { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
.feat-card  { background:var(--card); border-radius:12px; padding:28px 22px; border:1px solid var(--border); transition:transform .2s,box-shadow .2s,border-color .2s; }
.feat-card:hover { transform:translateY(-4px); box-shadow:0 12px 36px rgba(0,0,0,.35); border-color:rgba(220,38,38,.3); }
.feat-icon  { font-size:1.9rem; margin-bottom:16px; display:block; }
.feat-title { font-family:'Playfair Display',serif; font-size:1.08rem; font-weight:700; color:#111111; margin-bottom:10px; }
.feat-text  { color:var(--muted); font-size:.87rem; line-height:1.65; }

/* Как работает */
.how        { background:var(--bg); padding:90px 6%; border-top:1px solid var(--border); }
.steps      { display:grid; grid-template-columns:repeat(3,1fr); gap:40px; max-width:840px; position:relative; }
.steps::before {
  content:''; position:absolute; top:25px;
  left:calc(16.66% + 25px); right:calc(16.66% + 25px); height:1px;
  background-image:repeating-linear-gradient(90deg, var(--accent) 0, var(--accent) 6px, transparent 6px, transparent 14px);
}
.step-n { width:52px; height:52px; border-radius:50%; border:2px solid var(--accent); display:flex; align-items:center; justify-content:center; font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:700; color:var(--accent); margin-bottom:22px; background:var(--bg); position:relative; z-index:1; }
.step-t { font-family:'Playfair Display',serif; font-size:1.08rem; font-weight:700; color:#111111; margin-bottom:10px; }
.step-d { color:var(--muted); font-size:.87rem; line-height:1.65; }

/* Критерии */
.crit       { background:var(--bg); padding:80px 6%; border-top:1px solid var(--border); }
.crit-grid  { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-top:38px; }
.crit-tile  { background:var(--card); border-radius:8px; padding:16px 14px; border-left:3px solid var(--accent); border:1px solid var(--border); border-left:3px solid var(--accent); transition:box-shadow .15s,border-color .15s; }
.crit-tile:hover { box-shadow:0 4px 16px rgba(0,0,0,.3); border-color:rgba(220,38,38,.3); }
.ck  { font-family:'JetBrains Mono',monospace; font-size:.64rem; font-weight:700; color:var(--accent); margin-bottom:5px; letter-spacing:.06em; }
.cn  { font-size:.76rem; color:var(--text); line-height:1.4; font-weight:500; }
.cm  { margin-top:8px; font-family:'JetBrains Mono',monospace; font-size:.62rem; color:var(--muted); }

/* Нижний CTA */
.home-cta   { background:var(--bg); padding:80px 6%; border-top:1px solid var(--border); text-align:center; }
.home-footer{ background:var(--bg); padding:24px 6%; text-align:center; color:rgba(255,255,255,.2); font-size:.77rem; border-top:1px solid var(--border); }

/* ══════════ КНОПКИ ══════════ */
.stButton>button {
  font-family:'Inter',sans-serif !important;
  border-radius:6px !important; font-weight:600 !important;
  transition:all .15s !important; border:none !important; cursor:pointer !important;
  background:var(--accent) !important; color:#111111 !important;
  padding:12px 30px !important; font-size:.93rem !important; letter-spacing:.02em !important;
}
.stButton>button:hover {
  background:#B91C1C !important;
  box-shadow:0 6px 22px rgba(220,38,38,.3) !important;
  transform:translateY(-1px) !important;
}
[data-testid="stBaseButton-secondary"] {
  background:var(--card) !important;
  color:var(--muted) !important;
  border:1.5px solid var(--border) !important;
  padding:8px 16px !important;
  font-size:.82rem !important;
}
[data-testid="stBaseButton-secondary"]:hover {
  background:var(--bg2) !important;
  color:var(--text) !important;
  transform:none !important;
  box-shadow:none !important;
}
.stDownloadButton>button {
  background:var(--card) !important; color:var(--text) !important;
  border:1.5px solid var(--border) !important; border-radius:6px !important;
  font-family:'Inter',sans-serif !important; font-weight:600 !important;
  padding:11px 24px !important; font-size:.9rem !important; transition:all .15s !important;
}
.stDownloadButton>button:hover { background:var(--bg2) !important; border-color:var(--accent) !important; transform:translateY(-1px) !important; }

/* ══════════ СТРАНИЦА ПРОВЕРКИ ══════════ */
.ch-nav { background:var(--bg2); padding:16px 6%; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--border); }
.ch-logo { font-family:'Playfair Display',serif; font-weight:700; color:#111111; font-size:1.12rem; }
.ch-logo span { color:var(--accent); }
.ch-tagline { color:var(--muted); font-size:.78rem; }

/* Переключатель режима */
.mode-wrap { display:flex; align-items:center; gap:8px; margin-bottom:28px; }
.mode-label { color:var(--muted); font-size:.82rem; margin-right:6px; }

/* Метки полей */
.field-label { font-weight:600; font-size:.84rem; color:var(--text); margin-bottom:6px; }
.field-note  { font-size:.76rem; color:var(--muted); margin-top:4px; }

/* Текстовые области */
[data-testid="stTextArea"] label { display:none !important; }
[data-testid="stTextArea"] textarea {
  background:var(--card) !important; border:1.5px solid var(--border) !important;
  border-radius:6px !important; font-family:'Inter',sans-serif !important;
  font-size:.91rem !important; color:var(--text) !important; line-height:1.76 !important;
  transition:border-color .15s, box-shadow .15s !important;
}
[data-testid="stTextArea"] textarea:focus {
  border-color:var(--accent) !important;
  box-shadow:0 0 0 3px rgba(220,38,38,.12) !important; outline:none !important;
}
[data-testid="stTextArea"] textarea::placeholder { color:var(--muted) !important; opacity:.7 !important; }

/* Загрузчик файлов */
[data-testid="stFileUploaderDropzone"] {
  background:var(--card) !important; border:2px dashed var(--border) !important;
  border-radius:8px !important; transition:border-color .15s !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--accent) !important; }
[data-testid="stFileUploaderDropzone"] > div { color:var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] span  { color:var(--muted) !important; }

/* Блок транскрипции */
.transcr-note {
  background:var(--accent-d); border:1px solid rgba(220,38,38,.28);
  border-radius:8px; padding:14px 18px; margin-bottom:14px;
  font-size:.85rem; color:#FCA5A5; line-height:1.55;
  animation:fadeIn .4s ease both;
}
.transcr-note strong { color:#111111; }

/* Результаты */
.corr-card { background:var(--card); border:1.5px solid var(--border); border-left:4px solid var(--accent); border-radius:10px; padding:28px 26px; margin-bottom:34px; animation:fadeIn .45s ease both; }
.corr-hdr  { font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:700; color:#111111; margin-bottom:16px; display:flex; align-items:center; gap:10px; }
.corr-hdr::before { content:''; display:block; width:8px; height:8px; background:var(--accent); border-radius:50%; flex-shrink:0; }
.corr-body { font-size:.92rem; color:var(--text); line-height:1.82; }
.corr-body strong { color:var(--accent); background:var(--accent-d); padding:0 2px; border-radius:2px; text-decoration:underline wavy var(--accent); text-underline-offset:3px; font-weight:700; }

/* Карточки критериев */
.sc-grid  { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-bottom:34px; }
.sc-card  { background:var(--card); border-radius:10px; padding:18px 20px; border:1.5px solid var(--border); animation:slideCard .4s ease both; opacity:0; animation-fill-mode:both; }
.sc-card.sc-green { border-color:#238636; background:#0D2818; }
.sc-card.sc-amber { border-color:#9E6A03; background:#2B1D00; }
.sc-card.sc-red   { border-color:#DA3633; background:#2D0A0A; }
.sc-head  { display:flex; align-items:center; justify-content:space-between; margin-bottom:7px; }
.sc-k     { font-family:'JetBrains Mono',monospace; font-size:.67rem; font-weight:700; color:#111111; background:var(--accent); padding:3px 8px; border-radius:4px; letter-spacing:.05em; }
.sc-frac  { font-family:'JetBrains Mono',monospace; font-size:1.2rem; font-weight:700; }
.sc-green .sc-frac{color:#3FB950;} .sc-amber .sc-frac{color:#E3B341;} .sc-red .sc-frac{color:#F85149;}
.sc-name  { font-size:.82rem; font-weight:600; color:var(--text); margin-bottom:9px; }
.sc-dots  { display:flex; gap:5px; margin-bottom:8px; }
.dot      { width:10px; height:10px; border-radius:50%; border:1.5px solid var(--border); background:transparent; }
.sc-green .dot.filled{background:#238636;border-color:#238636;}
.sc-amber .dot.filled{background:#9E6A03;border-color:#9E6A03;}
.sc-red   .dot.filled{background:#DA3633;border-color:#DA3633;}
.sc-comment { font-size:.79rem; color:var(--muted); line-height:1.55; }

/* Итоговый балл */
.total-wrap  { display:flex; align-items:center; gap:24px; background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:26px 28px; margin-bottom:22px; animation:fadeIn .6s .5s ease both; opacity:0; animation-fill-mode:both; }
.total-stamp { width:86px; height:86px; border-radius:50%; border:3px solid var(--accent); display:flex; flex-direction:column; align-items:center; justify-content:center; color:var(--accent); flex-shrink:0; animation:stampIn .5s .65s ease both; opacity:0; animation-fill-mode:both; }
.total-num   { font-family:'JetBrains Mono',monospace; font-size:1.75rem; font-weight:700; line-height:1; }
.total-den   { font-size:.5rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; opacity:.7; }
.total-text h3 { font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:700; color:#111111; margin:0 0 6px; }
.total-text p  { color:var(--muted); font-size:.86rem; margin:0; line-height:1.55; }

/* Ошибка */
.err-card  { background:#2D0A0A; border:1.5px solid #DA3633; border-radius:8px; padding:20px 22px; animation:fadeIn .4s ease both; }
.err-title { color:#F85149; font-weight:700; font-size:.93rem; margin-bottom:6px; }
.err-text  { color:var(--muted); font-size:.84rem; line-height:1.55; }

/* Адаптив */
@media(max-width:820px){
  .hero-inner{grid-template-columns:1fr;} .paper{display:none;}
  .feat-grid,.steps,.crit-grid,.sc-grid{grid-template-columns:1fr !important;}
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Главная страница
# ══════════════════════════════════════════════════════════
def show_home():
    # Hero
    st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div>
      <div class="eyebrow">🎓 &thinsp;ЕГЭ по русскому языку &nbsp;·&nbsp; 2026</div>
      <h1 class="hero-h1">
        Проверка сочинения<br>
        с точностью <span class="accent">эксперта</span>
      </h1>
      <p class="hero-sub">
        ИИ анализирует сочинение по всем 10 критериям ФИПИ, находит ошибки
        и объясняет, как повысить балл. Поддерживается текст и фото рукописи.
      </p>
      <div class="hero-badges">
        <div class="hbadge"><div class="hbadge-dot"></div>К1–К10 · ФИПИ 2026</div>
        <div class="hbadge"><div class="hbadge-dot"></div>Максимум 22 балла</div>
        <div class="hbadge"><div class="hbadge-dot"></div>Текст и фото рукописи</div>
      </div>
    </div>
    <div>
      <div class="paper">
        <div class="paper-h">Сочинение-рассуждение</div>
        <div class="paper-t">
          В своём тексте автор <span class="e">поднемает</span> проблему отношения к искусству.
          Позиция автора состоит в том, что творчество не нуждается
          в <span class="e">оправданием</span>. Я <span class="e">соглашаюсь</span>
          с данной точкой зрения...
        </div>
        <div class="pscores">
          <div class="ps g"><span class="pk">К1</span><span class="pv">1</span></div>
          <div class="ps g"><span class="pk">К2</span><span class="pv">3</span></div>
          <div class="ps a"><span class="pk">К3</span><span class="pv">1</span></div>
          <div class="ps a"><span class="pk">К7</span><span class="pv">1</span></div>
          <div class="ps r"><span class="pk">К8</span><span class="pv">0</span></div>
        </div>
        <div class="paper-stamp"><div class="ps-n">17</div><div class="ps-d">из 22</div></div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # CTA кнопка
    st.markdown("""
<div style="background:var(--bg);padding:64px 6% 0;text-align:center;border-top:1px solid var(--border)">
  <p style="font-family:'Playfair Display',serif;font-size:clamp(1.3rem,2vw,1.8rem);color:#111111;margin:0 0 8px;font-weight:700">
    Готовы проверить своё сочинение?
  </p>
  <p style="color:var(--muted);font-size:.85rem;margin:0 0 30px">
    Бесплатно · Без регистрации · Текст или фото рукописи
  </p>
</div>
""", unsafe_allow_html=True)
    _, c, _ = st.columns([3, 2, 3])
    with c:
        if st.button("Начать проверку →", use_container_width=True, key="cta_top"):
            st.session_state.page = "check"
            st.session_state.result = None
            st.rerun()

    # Возможности
    st.markdown("""
<div class="features">
  <div class="sec">
    <div class="sec-label">Возможности</div>
    <div class="sec-title">Что умеет ИИ-проверка</div>
    <div class="sec-sub">Полный анализ по официальной шкале ФИПИ — так же, как это делает реальный эксперт ЕГЭ.</div>
    <div class="feat-grid">
      <div class="feat-card">
        <span class="feat-icon">📷</span>
        <div class="feat-title">Читает рукописи</div>
        <div class="feat-text">Загрузите фотографию сочинения — ИИ расшифрует рукописный текст с сохранением знаков препинания, вы проверите расшифровку и запустите анализ.</div>
      </div>
      <div class="feat-card">
        <span class="feat-icon">📊</span>
        <div class="feat-title">Ставит баллы по К1–К10</div>
        <div class="feat-text">Каждый из 10 критериев оценивается отдельно с кратким пояснением — вы видите, за что именно снижен балл.</div>
      </div>
      <div class="feat-card">
        <span class="feat-icon">🔍</span>
        <div class="feat-title">Выделяет ошибки</div>
        <div class="feat-text">Орфографические, пунктуационные, грамматические и речевые ошибки подсвечиваются прямо в тексте сочинения.</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Как работает (фото)
    st.markdown("""
<div class="how">
  <div class="sec">
    <div class="sec-label">Проверка рукописи</div>
    <div class="sec-title">Два шага для фото</div>
    <div class="sec-sub">ИИ сначала расшифрует почерк, вы проверите результат, а затем запустите полноценную проверку.</div>
    <div class="steps">
      <div>
        <div class="step-n">01</div>
        <div class="step-t">Загрузите фото</div>
        <div class="step-d">Сфотографируйте рукописное сочинение и загрузите его вместе с исходным текстом.</div>
      </div>
      <div>
        <div class="step-n">02</div>
        <div class="step-t">Проверьте расшифровку</div>
        <div class="step-d">ИИ распознает рукопись и покажет текст. Вы можете исправить ошибки распознавания — особенно запятые и точки.</div>
      </div>
      <div>
        <div class="step-n">03</div>
        <div class="step-t">Получите оценку</div>
        <div class="step-d">Нажмите «Проверить» — ИИ выставит баллы по К1–К10 и объяснит каждый критерий.</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Критерии
    tiles = "".join(
        f'<div class="crit-tile"><div class="ck">{k}</div><div class="cn">{name}</div><div class="cm">макс. {ms} б.</div></div>'
        for k, (name, ms) in CRITERIA.items()
    )
    st.markdown(f"""
<div class="crit">
  <div class="sec">
    <div class="sec-label">Критерии ФИПИ 2026</div>
    <div class="sec-title">10 критериев · 22 балла</div>
    <div class="sec-sub">Оценка по официальной шкале — той же, что используют реальные эксперты ЕГЭ.</div>
    <div class="crit-grid">{tiles}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Нижний CTA
    st.markdown("""
<div class="home-cta">
  <p style="font-family:'Playfair Display',serif;font-size:clamp(1.7rem,2.8vw,2.3rem);font-weight:700;color:#111111;margin:0 0 12px">
    Начните прямо сейчас
  </p>
  <p style="color:var(--muted);font-size:.95rem;margin:0 0 32px">
    Вставьте текст или загрузите фото рукописи — результат за секунды
  </p>
</div>
""", unsafe_allow_html=True)
    _, c2, _ = st.columns([3, 2, 3])
    with c2:
        if st.button("Проверить сочинение →", use_container_width=True, key="cta_bot"):
            st.session_state.page = "check"
            st.session_state.result = None
            st.rerun()

    st.markdown('<div class="home-footer">ЕГЭ Эксперт · Проверка сочинений на основе критериев ФИПИ 2026</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Страница проверки
# ══════════════════════════════════════════════════════════
def show_checker():
    # Навбар
    st.markdown("""
<div class="ch-nav">
  <div class="ch-logo">ЕГЭ <span>Эксперт</span></div>
  <div class="ch-tagline">Проверка сочинений · ФИПИ 2026</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="padding:14px 6% 0">', unsafe_allow_html=True)
    if st.button("← На главную", type="secondary", key="back"):
        st.session_state.page = "home"
        st.session_state.result = None
        st.session_state.transcription = ""
        st.session_state.transcription_done = False
        if "transcribed" in st.session_state:
            del st.session_state["transcribed"]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Заголовок
    st.markdown("""
<div style="padding:28px 6% 0;max-width:1200px;margin:0 auto">
  <h1 style="font-family:'Playfair Display',serif;font-size:clamp(1.5rem,2.3vw,2rem);font-weight:700;color:#111111;margin:0 0 6px">
    Проверка сочинения
  </h1>
  <p style="color:var(--muted);font-size:.89rem;margin:0 0 24px">
    Выберите режим: вставьте текст или загрузите фотографию рукописи
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Переключатель режима ──────────────────────────────────
    st.markdown('<div style="padding:0 6%;max-width:1200px;margin:0 auto">', unsafe_allow_html=True)
    mc1, mc2, _ = st.columns([1.4, 1.6, 8])
    with mc1:
        if st.button(
            "📝  Вставить текст",
            type="primary" if st.session_state.input_mode == "text" else "secondary",
            use_container_width=True,
            key="mode_text",
        ):
            st.session_state.input_mode = "text"
            st.rerun()
    with mc2:
        if st.button(
            "📷  Загрузить фото рукописи",
            type="primary" if st.session_state.input_mode == "photo" else "secondary",
            use_container_width=True,
            key="mode_photo",
        ):
            st.session_state.input_mode = "photo"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # ════════════════════════════════
    #  РЕЖИМ: ТЕКСТ
    # ════════════════════════════════
    if st.session_state.input_mode == "text":
        st.markdown('<div style="padding:0 6%;max-width:1200px;margin:0 auto">', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="field-label">📄 Исходный текст</div>', unsafe_allow_html=True)
            source_text = st.text_area(
                "src_h", height=300, label_visibility="collapsed",
                placeholder="Вставьте исходный текст здесь...", key="src"
            )
        with col2:
            st.markdown('<div class="field-label">✏️ Сочинение ученика</div>', unsafe_allow_html=True)
            essay_text = st.text_area(
                "essay_h", height=300, label_visibility="collapsed",
                placeholder="Вставьте текст сочинения здесь...", key="essay"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        _, bc, _ = st.columns([2.5, 1, 2.5])
        with bc:
            check_clicked = st.button("🔍 Проверить сочинение", use_container_width=True, key="check_text")

        if check_clicked:
            if not source_text.strip() or not essay_text.strip():
                st.warning("Заполните оба поля — исходный текст и сочинение.")
            elif "OPENROUTER_API_KEY" not in st.secrets:
                st.error("Не найден ключ OPENROUTER_API_KEY в st.secrets.")
            else:
                with st.spinner("ИИ анализирует сочинение по критериям ФИПИ..."):
                    try:
                        raw = call_api(source_text, essay_text)
                        st.session_state.raw = raw
                        st.session_state.result = extract_json(raw)
                    except Exception as e:
                        st.error(f"Ошибка API: {e}")
                        st.session_state.result = None

    # ════════════════════════════════
    #  РЕЖИМ: ФОТО РУКОПИСИ
    # ════════════════════════════════
    else:
        st.markdown('<div style="padding:0 6%;max-width:1200px;margin:0 auto">', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="field-label">📄 Исходный текст (для сверки)</div>', unsafe_allow_html=True)
            source_photo = st.text_area(
                "src_ph", height=300, label_visibility="collapsed",
                placeholder="Вставьте исходный текст сюда...", key="src_photo"
            )

        with col2:
            st.markdown('<div class="field-label">📷 Фотография рукописного сочинения</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "photo_upload", label_visibility="collapsed",
                type=["jpg", "jpeg", "png", "webp", "heic"],
                key="photo_file",
            )
            if uploaded:
                st.image(uploaded, use_container_width=True)
                st.markdown(
                    '<div class="field-note">Убедитесь, что текст чёткий и хорошо освещён</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Шаг 1: распознать рукопись ──
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        _, bc1, _ = st.columns([2.5, 1.4, 2.5])
        with bc1:
            transcribe_clicked = st.button(
                "🔤 Распознать рукопись",
                use_container_width=True,
                key="transcribe_btn",
                disabled=not uploaded,
            )

        if transcribe_clicked:
            if not uploaded:
                st.warning("Сначала загрузите фотографию сочинения.")
            elif "OPENROUTER_API_KEY" not in st.secrets:
                st.error("Не найден ключ OPENROUTER_API_KEY в st.secrets.")
            else:
                with st.spinner("Читаю рукопись... Это может занять 10–30 секунд."):
                    try:
                        img_bytes = uploaded.read()
                        img_mime = uploaded.type or "image/jpeg"
                        transcription = transcribe_image(img_bytes, img_mime)
                        st.session_state.transcription = transcription
                        st.session_state.transcription_done = True
                        st.session_state.result = None
                        # сбросить редактируемую область, чтобы подставить новый текст
                        if "transcribed" in st.session_state:
                            del st.session_state["transcribed"]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка распознавания: {e}")

        # ── Шаг 2: редактирование и проверка ──
        if st.session_state.transcription_done and st.session_state.transcription:
            st.markdown('<div style="padding:0 6%;max-width:1200px;margin:0 auto">', unsafe_allow_html=True)
            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

            st.markdown("""
<div class="transcr-note">
  <strong>Шаг 2 — проверьте расшифровку перед отправкой.</strong><br>
  ИИ старается сохранить запятые и точки, но может ошибиться в неразборчивых местах.
  Отредактируйте текст ниже при необходимости, а затем нажмите «Проверить сочинение».
</div>
""", unsafe_allow_html=True)

            st.markdown('<div class="field-label">📄 Оригинальная расшифровка ИИ</div>', unsafe_allow_html=True)
            st.text_area(
                "orig_transcription",
                value=st.session_state.transcription,
                height=180,
                disabled=True,
                label_visibility="collapsed",
            )

            st.markdown('<div class="field-label">📝 Текст для проверки (можно редактировать)</div>', unsafe_allow_html=True)
            essay_text_photo = st.text_area(
                "transcribed_label",
                value=st.session_state.get("transcribed", st.session_state.transcription),
                height=320,
                label_visibility="collapsed",
                key="transcribed",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            confirmed = st.checkbox(
                "Я проверил распознанный текст и подтверждаю его корректность"
            )

            _, bc2, _ = st.columns([2.5, 1, 2.5])
            with bc2:
                check_photo = st.button(
                    "🔍 Проверить сочинение",
                    use_container_width=True,
                    key="check_photo",
                    disabled=not confirmed
                )

            if check_photo:
                src = st.session_state.get("src_photo", "")
                essay = st.session_state.get("transcribed", "")
                if not src.strip() or not essay.strip():
                    st.warning("Заполните исходный текст и убедитесь, что расшифровка не пустая.")
                elif "OPENROUTER_API_KEY" not in st.secrets:
                    st.error("Не найден ключ OPENROUTER_API_KEY в st.secrets.")
                else:
                    with st.spinner("ИИ анализирует сочинение по критериям ФИПИ..."):
                        try:
                            raw = call_api(src, essay)
                            st.session_state.raw = raw
                            st.session_state.result = extract_json(raw)
                        except Exception as e:
                            st.error(f"Ошибка API: {e}")
                            st.session_state.result = None

    # ════════════════════════════════
    #  РЕЗУЛЬТАТЫ (общие для обоих режимов)
    # ════════════════════════════════
    result = st.session_state.result

    if result:
        st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:36px 0">', unsafe_allow_html=True)
        st.markdown('<div style="padding:0 6% 60px;max-width:1200px;margin:0 auto">', unsafe_allow_html=True)

        # Исправленный текст
        corrected = str(result.get("corrected_text", "")).strip()
        if corrected:
            st.markdown(f"""
<div class="corr-card">
  <div class="corr-hdr">Сочинение с исправлениями</div>
  <div class="corr-body">{md_bold_to_html(corrected)}</div>
</div>""", unsafe_allow_html=True)

        # Оценка
        st.markdown("""
<p style="font-family:'Playfair Display',serif;font-size:1.33rem;font-weight:700;color:#111111;margin:0 0 6px">Оценка по критериям</p>
<p style="color:var(--muted);font-size:.87rem;margin:0 0 22px">К1–К10 · ФИПИ 2026 · Максимум 22 балла</p>
""", unsafe_allow_html=True)
        st.markdown(build_score_cards(result), unsafe_allow_html=True)

        # Итоговый балл
        scores = result.get("scores", {}) or {}
        total = 0
        for k, (_, ms) in CRITERIA.items():
            try:
                total += max(0, min(int(scores.get(k, 0)), ms))
            except (TypeError, ValueError):
                pass
        st.markdown(build_total(total), unsafe_allow_html=True)

        # Скачать отчёт
        src_for_report  = st.session_state.get("src", "") or st.session_state.get("src_photo", "")
        essay_for_report = st.session_state.get("essay", "") or st.session_state.get("transcribed", "")
        st.download_button(
            "📥 Скачать отчёт (.txt)",
            data=build_report(src_for_report, essay_for_report, result, total),
            file_name=f"ege_{datetime.now().strftime('%d%m%Y_%H%M')}.txt",
            mime="text/plain",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.raw and result is None:
        st.markdown('<div style="padding:0 6%;max-width:1200px;margin:0 auto">', unsafe_allow_html=True)
        st.markdown("""
<div class="err-card">
  <div class="err-title">⚠️ Не удалось разобрать ответ модели</div>
  <div class="err-text">
    Модель вернула ответ не в формате JSON. Попробуйте нажать «Проверить» ещё раз
    или укажите другую модель в OPENROUTER_MODEL в secrets.toml.
  </div>
</div>""", unsafe_allow_html=True)
        with st.expander("Показать необработанный ответ модели"):
            st.code(st.session_state.raw, language=None)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Роутер
# ══════════════════════════════════════════════════════════
inject_css()
if st.session_state.page == "home":
    show_home()
else:
    show_checker()
