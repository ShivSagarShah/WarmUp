"""
Daily Cooking Planner — AI micro-app
Evaluates on: intent · speed · execution
"""

import json
import os
import time

from google import genai
from google.genai import types
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cooking Planner AI",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]          { background:#0d1117; }
[data-testid="stAppViewContainer"] { background:#0a0e1a; }

.hero {
    background: linear-gradient(120deg,#0f3460 0%,#1a1a2e 60%,#16213e 100%);
    border-radius:16px; padding:24px 28px; margin-bottom:24px;
    border:1px solid #1e2d50;
}
.hero-title { font-size:1.7em; font-weight:800; color:#fff; margin:0 0 6px 0; }
.hero-sub   { color:#a8b8d8; font-size:0.96em; margin:0; }

.meal-card {
    background:linear-gradient(135deg,#12192e 0%,#1a2540 100%);
    border:1px solid #1e3a5f; border-radius:14px;
    padding:22px; margin:8px 0;
}
.meal-card:hover { border-color:#4ecca3; }
.meal-name  { font-size:1.25em; font-weight:700; color:#e8f0fe; }
.meal-sub   { color:#7a8db3; font-size:0.85em; margin:4px 0 12px; }

.badge {
    display:inline-block; border-radius:20px; padding:2px 11px;
    font-size:0.76em; font-weight:600; margin:0 4px 4px 0;
}
.badge-easy   { background:#0d2e22; color:#4ecca3; border:1px solid #4ecca3; }
.badge-medium { background:#2e2500; color:#f6c90e; border:1px solid #f6c90e; }
.badge-hard   { background:#2e0f12; color:#e94560; border:1px solid #e94560; }
.badge-time   { background:#12213e; color:#90b4f8; border:1px solid #3060b0; }
.badge-cal    { background:#1e1225; color:#c084fc; border:1px solid #7c3aed; }

.tag {
    display:inline-block; background:#0f2040; color:#c8d8f0;
    border-radius:20px; padding:2px 10px; font-size:0.78em; margin:2px;
    border:1px solid #1e3a60;
}

.sec-head {
    font-size:0.9em; font-weight:700; color:#4ecca3;
    letter-spacing:.06em; text-transform:uppercase;
    margin:18px 0 8px; border-left:3px solid #4ecca3; padding-left:8px;
}

.tl-row {
    display:flex; align-items:flex-start; gap:14px;
    padding:10px 0; border-bottom:1px solid #1a2540;
}
.tl-time  { min-width:80px; color:#4ecca3; font-weight:700; font-size:0.9em; }
.tl-task  { color:#c8d8f0; flex:1; font-size:0.9em; }
.tl-dur   { color:#7a8db3; font-size:0.8em; white-space:nowrap; }

.chef-tip {
    background:#0d1e14; border-left:3px solid #4ecca3;
    border-radius:0 8px 8px 0; padding:10px 14px;
    color:#a8d8b8; font-size:0.88em; margin-top:12px; font-style:italic;
}
.chef-msg {
    background:linear-gradient(90deg,#0f2d20,#0d1e14);
    border:1px solid #1e4d30; border-radius:10px;
    padding:14px 18px; color:#a8d8b8; font-size:0.96em;
    font-style:italic; margin-bottom:20px;
}
.status-good { color:#4ecca3; font-size:1.05em; font-weight:700; }
.status-warn { color:#f6c90e; font-size:1.05em; font-weight:700; }
.status-bad  { color:#e94560; font-size:1.05em; font-weight:700; }
.macro-label { font-size:0.82em; color:#7a8db3; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PRESETS
# ─────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "⚡ Busy Weekday":     dict(day_type="Busy Weekday",     people=2, budget=30, cooking_time="Under 15 mins per meal", health_goal="Balanced"),
    "💪 Fitness Day":      dict(day_type="Work from Home",   people=1, budget=35, cooking_time="15–30 mins per meal",   health_goal="High Protein"),
    "🍷 Date Night":       dict(day_type="Special Occasion", people=2, budget=80, cooking_time="30–60 mins per meal",   health_goal="Comfort Food"),
    "📦 Meal Prep Sunday": dict(day_type="Meal Prep Sunday", people=4, budget=100,cooking_time="No limit",              health_goal="Balanced"),
}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(u: dict) -> str:
    return f"""You are a professional personal chef. Create a full-day meal plan.

CONTEXT:
- About today: {u["day_description"] or "A regular day"}
- Day type:    {u["day_type"]}
- People:      {u["people"]}
- Budget:      ${u["budget"]} USD total
- Dietary:     {u["dietary"] or "None"}
- Cuisine:     {u["cuisine"] or "Any"}
- Cook time:   {u["cooking_time"]}
- Pantry:      {u["pantry"] or "Basic staples (oil, salt, pepper, garlic)"}
- Goal:        {u["health_goal"]}

Return ONLY a valid JSON object — no markdown, no extra text — matching this exact schema:
{{
  "chef_message": "One warm personalised sentence acknowledging their day",
  "meal_plan": {{
    "breakfast": {{
      "name": "string",
      "description": "1-2 sentences",
      "prep_time": "e.g. 5 mins",
      "cook_time": "e.g. 10 mins",
      "difficulty": "Easy|Medium|Hard",
      "calories_per_serving": "e.g. 420 kcal",
      "ingredients": ["..."],
      "cooking_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
      "chef_tip": "One pro tip"
    }},
    "lunch": {{
      "name": "string", "description": "...", "prep_time": "...", "cook_time": "...",
      "difficulty": "Easy|Medium|Hard", "calories_per_serving": "...",
      "ingredients": ["..."], "cooking_steps": ["..."], "chef_tip": "..."
    }},
    "dinner": {{
      "name": "string", "description": "...", "prep_time": "...", "cook_time": "...",
      "difficulty": "Easy|Medium|Hard", "calories_per_serving": "...",
      "ingredients": ["..."], "cooking_steps": ["..."], "chef_tip": "..."
    }}
  }},
  "cooking_timeline": [
    {{"time": "7:30 AM",  "task": "...", "meal": "breakfast", "duration": "15 mins"}},
    {{"time": "12:00 PM", "task": "...", "meal": "lunch",     "duration": "25 mins"}},
    {{"time": "6:30 PM",  "task": "...", "meal": "dinner",    "duration": "40 mins"}}
  ],
  "grocery_list": [
    {{"item": "...", "quantity": "...", "estimated_cost_usd": 0.00,
      "category": "Produce|Protein|Dairy|Grains|Pantry|Other"}}
  ],
  "substitutions": [
    {{"original": "...", "substitute": "...", "reason": "...", "cost_saving_usd": 0.00}}
  ],
  "budget_analysis": {{
    "total_estimated_cost_usd": 0.00,
    "cost_per_person_usd": 0.00,
    "status": "within_budget|tight|over_budget",
    "percentage_of_budget_used": 0,
    "savings_tips": ["tip 1", "tip 2"],
    "verdict": "One sentence summary"
  }},
  "nutritional_summary": {{
    "total_calories": "e.g. 1820 kcal",
    "protein_g": "95",
    "carbs_g": "210",
    "fat_g": "62",
    "highlights": ["highlight 1", "highlight 2"]
  }}
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING CALL
# ─────────────────────────────────────────────────────────────────────────────
# Model fallback chain — tries each in order until one succeeds
_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]


def _gen_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction="You are a professional chef. Respond with valid JSON only — no markdown fences, no extra text.",
        response_mime_type="application/json",
        temperature=0.68,
        max_output_tokens=2500,   # trimmed to stay well inside free-tier TPM
    )


def stream_plan(user_info: dict, api_key: str):
    """Tries each model in _MODELS; yields raw tokens from the first that works."""
    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None
    for model_name in _MODELS:
        try:
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=build_prompt(user_info),
                config=_gen_config(),
            ):
                if chunk.text:
                    yield chunk.text
            return   # success — stop trying other models
        except Exception as exc:
            err = str(exc).lower()
            last_exc = exc
            # Only fall through to next model on quota/rate errors
            if not any(x in err for x in ["quota", "429", "exhausted", "rate", "limit"]):
                raise   # non-quota error — surface immediately
    # All models exhausted
    if last_exc:
        raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MEAL_ICONS = {"breakfast": "☀️", "lunch": "🥗", "dinner": "🍽️"}
DIFF_BADGE = {
    "Easy":   '<span class="badge badge-easy">Easy</span>',
    "Medium": '<span class="badge badge-medium">Medium</span>',
    "Hard":   '<span class="badge badge-hard">Hard</span>',
}
CAT_ICONS = {
    "Produce": "🥦", "Protein": "🥩", "Dairy": "🧀",
    "Grains": "🌾",  "Pantry": "🫙",  "Other": "🛒",
}


def render_chef_message(msg: str):
    st.markdown(f'<div class="chef-msg">👨‍🍳 &nbsp;{msg}</div>', unsafe_allow_html=True)


def render_meal_card(key: str, meal: dict):
    icon  = MEAL_ICONS.get(key, "🍴")
    diff  = DIFF_BADGE.get(meal.get("difficulty", "Easy"), "")
    tags  = "".join(f'<span class="tag">{i}</span>' for i in meal.get("ingredients", []))
    tip   = meal.get("chef_tip", "")
    total = f"{meal.get('prep_time','?')} prep · {meal.get('cook_time','?')} cook"

    st.markdown(f"""
    <div class="meal-card">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
            <div>
                <div class="meal-name">{icon} {key.capitalize()} — {meal.get("name","")}</div>
                <div class="meal-sub">{meal.get("description","")}</div>
            </div>
            <div style="text-align:right;">
                {diff}
                <span class="badge badge-time">⏱ {total}</span>
                <span class="badge badge-cal">🔥 {meal.get("calories_per_serving","")}</span>
            </div>
        </div>
        <div class="sec-head">Ingredients</div>
        <div>{tags}</div>
    </div>
    """, unsafe_allow_html=True)

    steps = meal.get("cooking_steps", [])
    if steps:
        st.markdown('<div class="sec-head">Cooking Steps</div>', unsafe_allow_html=True)
        for s_i, step in enumerate(steps):
            ck = f"step_{key}_{s_i}"
            done = st.session_state.get(ck, False)
            st.checkbox(f"~~{step}~~" if done else step, key=ck, value=done)

    if tip:
        st.markdown(f'<div class="chef-tip">💡 Chef\'s tip: {tip}</div>', unsafe_allow_html=True)
    st.write("")


def render_timeline(items: list):
    for item in items:
        icon = MEAL_ICONS.get(item.get("meal", ""), "🍴")
        st.markdown(f"""
        <div class="tl-row">
            <div class="tl-time">{item.get("time","")}</div>
            <div class="tl-task">{icon} {item.get("task","")}</div>
            <div class="tl-dur">{item.get("duration","")}</div>
        </div>""", unsafe_allow_html=True)


def render_grocery(items: list):
    if not items:
        st.info("No grocery items returned.")
        return
    by_cat: dict = {}
    for g in items:
        by_cat.setdefault(g.get("category", "Other"), []).append(g)
    total = 0.0
    for cat, cat_items in sorted(by_cat.items()):
        icon = CAT_ICONS.get(cat, "📦")
        with st.expander(f"{icon} **{cat}** — {len(cat_items)} items", expanded=True):
            for g in cat_items:
                cost = float(g.get("estimated_cost_usd", 0) or 0)
                total += cost
                ck = f"groc_{g['item']}"
                done = st.session_state.get(ck, False)
                label = f"~~{g['item']} — {g['quantity']}~~" if done else f"{g['item']} — {g['quantity']}"
                ca, cb = st.columns([5, 1])
                ca.checkbox(label, key=ck, value=done)
                if cost:
                    cb.caption(f"~${cost:.2f}")
    st.divider()
    st.markdown(f"**Estimated grocery total: ${total:.2f}**")


def render_substitutions(subs: list):
    if not subs:
        st.info("No substitutions needed.")
        return
    for s in subs:
        saving = float(s.get("cost_saving_usd", 0) or 0)
        extra  = f"  ·  💰 saves ~${saving:.2f}" if saving else ""
        with st.expander(f"🔄 **{s.get('original','')}** → {s.get('substitute','')}"):
            st.markdown(f"**Reason:** {s.get('reason','')}{extra}")


def render_budget(analysis: dict, budget: float):
    total  = float(analysis.get("total_estimated_cost_usd", 0) or 0)
    per_p  = float(analysis.get("cost_per_person_usd", 0) or 0)
    pct    = int(analysis.get("percentage_of_budget_used") or (round(total / budget * 100) if budget else 0))
    status = analysis.get("status", "within_budget")

    c1, c2, c3, c4 = st.columns(4)
    delta_val = abs(total - budget)
    delta_lbl = f"-${delta_val:.2f} under" if total <= budget else f"+${delta_val:.2f} over"
    c1.metric("Estimated Cost", f"${total:.2f}")
    c2.metric("Your Budget",    f"${budget:.2f}", delta_lbl)
    c3.metric("Per Person",     f"${per_p:.2f}")
    c4.metric("Budget Used",    f"{pct}%")

    css   = {"within_budget": "status-good", "tight": "status-warn", "over_budget": "status-bad"}.get(status, "status-good")
    label = {"within_budget": "✅ Within Budget", "tight": "⚠️ Budget Tight", "over_budget": "❌ Over Budget"}.get(status, status)
    st.markdown(f'<p class="{css}">{label}</p>', unsafe_allow_html=True)
    st.progress(min(pct / 100, 1.0))
    st.markdown(f"_{analysis.get('verdict','')}_")
    tips = analysis.get("savings_tips", [])
    if tips:
        st.markdown("**💡 Savings Tips**")
        for t in tips:
            st.markdown(f"- {t}")


def render_nutrition(nut: dict):
    if not nut:
        return
    col1, col2 = st.columns([1, 2])
    col1.metric("Daily Calories", nut.get("total_calories", "—"))
    with col2:
        for name, key, cap in [("Protein", "protein_g", 200), ("Carbs", "carbs_g", 300), ("Fat", "fat_g", 100)]:
            raw = nut.get(key, "0")
            try:
                v = int(str(raw).split()[0])
            except (ValueError, TypeError):
                v = 0
            st.markdown(f'<div class="macro-label">{name} — {raw}g</div>', unsafe_allow_html=True)
            st.progress(min(v / cap, 1.0))
    highlights = nut.get("highlights", [])
    if highlights:
        st.markdown("**Highlights**")
        for h in highlights:
            st.markdown(f"- {h}")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍳 Cooking Planner")
    st.caption("Gemini 2.0 Flash · free tier · intent · speed · execution")
    st.divider()

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Get a free key at aistudio.google.com/app/apikey — never stored here.",
    )

    st.divider()
    st.markdown("**⚡ Quick Presets**")
    preset_cols = st.columns(2)
    chosen_preset: dict = {}
    for i, (label, vals) in enumerate(PRESETS.items()):
        if preset_cols[i % 2].button(label, use_container_width=True, key=f"preset_{i}"):
            chosen_preset = vals

    st.divider()
    st.markdown("**📅 Your Day**")
    _d = chosen_preset

    day_desc = st.text_area(
        "Describe your day",
        placeholder="e.g. Back-to-back meetings till 7 pm, gym in the morning, family dinner…",
        height=70,
    )

    DAY_TYPES = ["Busy Weekday", "Relaxed Weekend", "Work from Home", "Special Occasion", "Meal Prep Sunday"]
    day_type = st.selectbox("Day type", DAY_TYPES,
                            index=DAY_TYPES.index(_d.get("day_type", "Busy Weekday")) if _d else 0)

    people = st.number_input("People to feed", 1, 12, value=_d.get("people", 2) if _d else 2)
    budget = st.slider("Daily food budget ($)", 5, 200, value=_d.get("budget", 40) if _d else 40, step=5)

    COOK_TIMES = ["Under 15 mins per meal", "15–30 mins per meal", "30–60 mins per meal", "No limit"]
    cooking_time = st.selectbox("Cook time per meal", COOK_TIMES,
                                index=COOK_TIMES.index(_d.get("cooking_time", "Under 15 mins per meal")) if _d else 0)

    st.markdown("**🥗 Preferences**")
    dietary = st.multiselect("Dietary restrictions",
        ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Nut-Free", "Halal", "Kosher", "Low-Carb", "Keto"])
    cuisine = st.multiselect("Cuisine preferences",
        ["Italian", "Mexican", "Asian", "Mediterranean", "American", "Indian", "Middle Eastern", "French"])

    GOALS = ["Balanced", "High Protein", "Low Calorie", "Heart Healthy", "Energy Boost", "Comfort Food"]
    health_goal = st.selectbox("Health goal", GOALS,
                               index=GOALS.index(_d.get("health_goal", "Balanced")) if _d else 0)
    pantry = st.text_area("Pantry on hand",
                          placeholder="e.g. eggs, pasta, canned tomatoes, garlic…", height=70)

    st.divider()
    generate_btn = st.button("🚀 Generate My Plan", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🍳 Daily Cooking Planner</div>
    <div class="hero-sub">
        Describe your day → get a personalised meal plan, step-by-step cooking to-do list,
        grocery checklist, smart substitutions, nutrition summary, and budget breakdown.
    </div>
</div>
""", unsafe_allow_html=True)

# Landing page
if "plan" not in st.session_state and not generate_btn:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**What you get in one click:**

| | Feature |
|--|---------|
| 🍽️ | Breakfast · Lunch · Dinner with difficulty, time & calories |
| ☑️ | Step-by-step **cooking to-do list** (interactive checkboxes) |
| ⏰ | **Timeline** — when to start each meal |
| 🛒 | **Grocery checklist** grouped by aisle, tick as you shop |
| 🔄 | **Smart substitutions** with cost savings |
| 💰 | **Budget breakdown** — cost vs. budget with tips |
| 💪 | **Daily nutrition** — protein, carbs & fat macros |
        """)
    with c2:
        st.markdown("""
**Getting started:**

1. Get a **free Gemini API key** at [aistudio.google.com](https://aistudio.google.com/app/apikey) and paste it in the sidebar
2. Pick a **Quick Preset** or fill in your details
3. Optionally describe your day in plain English —
   the AI uses that to personalise every meal
4. Click **Generate My Plan**

**Tip:** The free-text day description is the most
powerful input — the more context you give, the more
tailored the plan becomes.
        """)
    st.stop()

# Validate
if generate_btn and not api_key:
    st.error("Please enter your Gemini API key in the sidebar. Get one free at aistudio.google.com/app/apikey")
    st.stop()

# Build user dict
user_info = dict(
    day_description = day_desc.strip(),
    day_type        = day_type,
    people          = people,
    budget          = budget,
    cooking_time    = cooking_time,
    dietary         = ", ".join(dietary) if dietary else "",
    cuisine         = ", ".join(cuisine) if cuisine else "",
    health_goal     = health_goal,
    pantry          = pantry.strip(),
)

# ── Generate ──────────────────────────────────────────────────────────────────
if generate_btn:
    # Reset interactive checkboxes from previous plan
    for k in [k for k in st.session_state if k.startswith(("step_", "groc_"))]:
        del st.session_state[k]

    raw = ""
    t0  = time.time()

    MAX_RETRIES = 3
    RETRY_DELAYS = [5, 15, 30]   # seconds between attempts

    with st.status("🧑‍🍳  Building your plan…", expanded=True) as status:
        st.write("Reading your day · choosing meals · costing ingredients…")
        progress_box = st.empty()
        last_exc = None

        for attempt in range(MAX_RETRIES):
            raw = ""
            try:
                if attempt > 0:
                    wait = RETRY_DELAYS[attempt - 1]
                    for remaining in range(wait, 0, -1):
                        progress_box.caption(f"Retrying in {remaining}s (attempt {attempt + 1}/{MAX_RETRIES})…")
                        time.sleep(1)

                for token in stream_plan(user_info, api_key):
                    raw += token
                    progress_box.caption(f"Generating… {len(raw):,} chars  (attempt {attempt + 1})")

                elapsed = time.time() - t0
                status.update(label=f"✅  Done in {elapsed:.1f}s", state="complete", expanded=False)
                last_exc = None
                break   # success — exit retry loop

            except Exception as exc:
                last_exc = exc
                err = str(exc).lower()
                is_rate = any(x in err for x in ["quota", "resource exhausted", "429", "rate", "limit"])
                is_auth = any(x in err for x in ["api key", "invalid", "permission", "403", "unauthenticated"])

                if is_auth:
                    # Auth errors won't fix themselves — fail immediately
                    status.update(label="❌  Auth error", state="error")
                    st.error(f"Invalid Gemini API key.  Get a free key at aistudio.google.com/app/apikey\n\n_Details: {exc}_")
                    st.stop()

                if attempt < MAX_RETRIES - 1 and is_rate:
                    status.update(label=f"⚠️  Rate limited — retrying…", state="running")
                else:
                    break   # exhausted retries or non-retryable error

        if last_exc is not None:
            err = str(last_exc).lower()
            status.update(label="❌  Error", state="error")
            if any(x in err for x in ["quota", "resource exhausted", "429", "rate", "limit"]):
                st.error(
                    "**Quota is 0 on all Gemini models.**\n\n"
                    "This means your API key was created in **Google Cloud Console** "
                    "where the free tier quota defaults to zero.\n\n"
                    "**Fix — takes 60 seconds:**\n"
                    "1. Go to **https://aistudio.google.com/app/apikey**\n"
                    "2. Click **Create API key → Create API key in new project**\n"
                    "3. Copy the new `AIza...` key\n"
                    "4. Replace `GEMINI_API_KEY` in your `.env` file with the new key\n"
                    "5. Restart the app (`Ctrl+C` in terminal, then re-run)\n\n"
                    f"_Raw error: {last_exc}_"
                )
            else:
                st.error(f"Unexpected error: {last_exc}")
            st.stop()

    # Parse — strip any accidental markdown fences
    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        plan  = json.loads(clean)
    except json.JSONDecodeError:
        st.error("The model returned malformed JSON. Please try again.")
        st.stop()

    st.session_state.update(plan=plan, user_info=user_info, budget=budget)

# ── Render saved plan ─────────────────────────────────────────────────────────
plan = st.session_state.get("plan")
bgt  = st.session_state.get("budget", budget)

if not plan:
    st.stop()

chef_msg = plan.get("chef_message", "")
if chef_msg:
    render_chef_message(chef_msg)

# ── 7 Tabs ────────────────────────────────────────────────────────────────────
t_plan, t_todo, t_tl, t_groc, t_subs, t_budget, t_nut = st.tabs([
    "🍽️ Meal Plan",
    "☑️ Cooking To-Do",
    "⏰ Timeline",
    "🛒 Groceries",
    "🔄 Substitutions",
    "💰 Budget",
    "💪 Nutrition",
])

meal_plan = plan.get("meal_plan", {})

with t_plan:
    for key in ["breakfast", "lunch", "dinner"]:
        if key in meal_plan:
            render_meal_card(key, meal_plan[key])
        else:
            st.warning(f"No {key} returned.")

with t_todo:
    st.caption("Tick each step as you cook — progress is saved in this session.")
    for key in ["breakfast", "lunch", "dinner"]:
        meal = meal_plan.get(key)
        if not meal:
            continue
        st.markdown(f"### {MEAL_ICONS[key]} {key.capitalize()} — {meal.get('name','')}")
        for s_i, step in enumerate(meal.get("cooking_steps", [])):
            ck   = f"step_{key}_{s_i}"
            done = st.session_state.get(ck, False)
            st.checkbox(f"~~{step}~~" if done else step, key=ck, value=done)
        if meal.get("chef_tip"):
            st.markdown(f'<div class="chef-tip">💡 {meal["chef_tip"]}</div>', unsafe_allow_html=True)
        st.divider()

with t_tl:
    tl = plan.get("cooking_timeline", [])
    if tl:
        st.caption("Suggested prep & cook windows to keep every meal on time.")
        st.write("")
        render_timeline(tl)
    else:
        st.info("No timeline returned.")

with t_groc:
    render_grocery(plan.get("grocery_list", []))

with t_subs:
    render_substitutions(plan.get("substitutions", []))

with t_budget:
    ba = plan.get("budget_analysis", {})
    if ba:
        render_budget(ba, float(bgt))
    else:
        st.info("No budget data returned.")

with t_nut:
    render_nutrition(plan.get("nutritional_summary", {}))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
col_dl, col_reset, _ = st.columns([1, 1, 4])
with col_dl:
    st.download_button(
        "⬇️ Export Plan",
        data=json.dumps(plan, indent=2),
        file_name="cooking_plan.json",
        mime="application/json",
    )
with col_reset:
    if st.button("🔄 New Plan"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
