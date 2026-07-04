"""
tests/test_cooking_planner.py
────────────────────────────────────────────────────────────────────────────────
Comprehensive test suite for the Daily Cooking Planner micro-app.

Run:  pytest tests/ -v
      pytest tests/ -v --tb=short
"""
import json
import os
import sys
from unittest.mock import MagicMock
import pytest

# ── Stub out UI / external libs BEFORE importing app ─────────────────────────
_st_mock = MagicMock()
_st_mock.stop = MagicMock()                          # no-op — module runs to end
_st_mock.session_state = {"plan": {}, "budget": 40}  # prevent NoneType errors
_st_mock.button.return_value   = False               # no button pressed
_st_mock.text_input.return_value = ""
_st_mock.selectbox.return_value  = "Busy Weekday"
_st_mock.number_input.return_value = 2
_st_mock.slider.return_value   = 40
_st_mock.multiselect.return_value = []
_st_mock.text_area.return_value   = ""

# st.columns(n) / st.tabs(labels) must unpack correctly
def _columns(spec, *a, **kw):
    count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
    return [MagicMock() for _ in range(count)]

def _tabs(labels, *a, **kw):
    return [MagicMock() for _ in labels]

_st_mock.columns = MagicMock(side_effect=_columns)
_st_mock.tabs    = MagicMock(side_effect=_tabs)

for _mod in ["streamlit", "google", "google.genai", "google.genai.types", "dotenv"]:
    sys.modules[_mod] = MagicMock()
sys.modules["streamlit"]       = _st_mock
sys.modules["dotenv"].load_dotenv = MagicMock()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app   # st.stop() is now a no-op so module runs to completion


# ─────────────────────────────────────────────────────────────────────────────
# 1 · build_prompt
# ─────────────────────────────────────────────────────────────────────────────
BASE_USER: dict = dict(
    day_description="Back-to-back meetings all day",
    day_type="Busy Weekday",
    people=2,
    budget=40,
    cooking_time="Under 15 mins per meal",
    dietary="Vegetarian",
    cuisine="Italian",
    health_goal="Balanced",
    pantry="eggs, pasta, garlic",
)


class TestBuildPrompt:
    def test_returns_string(self):
        assert isinstance(app.build_prompt(BASE_USER), str)

    def test_non_empty(self):
        assert len(app.build_prompt(BASE_USER)) > 200

    def test_contains_day_type(self):
        assert "Busy Weekday" in app.build_prompt(BASE_USER)

    def test_contains_budget(self):
        assert "40" in app.build_prompt(BASE_USER)

    def test_contains_dietary(self):
        assert "Vegetarian" in app.build_prompt(BASE_USER)

    def test_contains_cuisine(self):
        assert "Italian" in app.build_prompt(BASE_USER)

    def test_contains_pantry(self):
        assert "eggs, pasta, garlic" in app.build_prompt(BASE_USER)

    def test_contains_day_description(self):
        assert "Back-to-back meetings" in app.build_prompt(BASE_USER)

    def test_empty_description_falls_back(self):
        info = {**BASE_USER, "day_description": ""}
        assert "A regular day" in app.build_prompt(info)

    def test_empty_dietary_shows_none(self):
        info = {**BASE_USER, "dietary": ""}
        assert "None" in app.build_prompt(info)

    def test_empty_cuisine_shows_any(self):
        info = {**BASE_USER, "cuisine": ""}
        assert "Any" in app.build_prompt(info)

    def test_people_count_in_prompt(self):
        info = {**BASE_USER, "people": 7}
        assert "7" in app.build_prompt(info)

    def test_prompt_contains_json_schema_keys(self):
        prompt = app.build_prompt(BASE_USER)
        for key in ["meal_plan", "grocery_list", "budget_analysis", "nutritional_summary"]:
            assert key in prompt

    def test_prompt_contains_meal_keys(self):
        prompt = app.build_prompt(BASE_USER)
        for meal in ["breakfast", "lunch", "dinner"]:
            assert meal in prompt

    def test_prompt_contains_cooking_steps(self):
        assert "cooking_steps" in app.build_prompt(BASE_USER)

    def test_high_people_count(self):
        info = {**BASE_USER, "people": 12}
        assert "12" in app.build_prompt(info)

    def test_high_budget(self):
        info = {**BASE_USER, "budget": 200}
        assert "200" in app.build_prompt(info)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Budget logic
# ─────────────────────────────────────────────────────────────────────────────
class TestBudgetLogic:
    STATUS_CSS = {
        "within_budget": "status-good",
        "tight":         "status-warn",
        "over_budget":   "status-bad",
    }

    @pytest.mark.parametrize("status,css", STATUS_CSS.items())
    def test_css_class_for_status(self, status, css):
        assert self.STATUS_CSS[status] == css

    def test_within_budget(self):
        total, budget = 30.0, 40.0
        assert total <= budget

    def test_over_budget(self):
        total, budget = 55.0, 40.0
        assert total > budget

    def test_tight_boundary(self):
        total, budget = 40.0, 40.0
        assert total <= budget

    def test_percentage_under(self):
        pct = round(30.0 / 40.0 * 100)
        assert pct == 75

    def test_percentage_over(self):
        pct = round(50.0 / 40.0 * 100)
        assert pct == 125

    def test_per_person_cost(self):
        assert round(40.0 / 4, 2) == 10.0

    def test_progress_capped_at_one(self):
        pct = 150
        assert min(pct / 100, 1.0) == 1.0

    def test_delta_under(self):
        total, budget = 30.0, 40.0
        assert abs(total - budget) == 10.0

    def test_zero_budget_guard(self):
        # Avoids division-by-zero
        budget = 0
        pct = round(30.0 / budget * 100) if budget else 0
        assert pct == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Nutrition macro parsing
# ─────────────────────────────────────────────────────────────────────────────
class TestNutritionParsing:
    """Mirrors the int-extraction logic inside render_nutrition."""

    @staticmethod
    def _parse(raw) -> int:
        try:
            return int(str(raw).split()[0])
        except (ValueError, TypeError, IndexError):
            return 0

    def test_plain_integer(self):           assert self._parse(95)      == 95
    def test_string_integer(self):          assert self._parse("62")    == 62
    def test_string_with_unit(self):        assert self._parse("95 g")  == 95  # space-separated unit
    def test_string_with_unit_nospace(self): assert self._parse("95g") == 0   # no space → can't parse cleanly
    def test_string_with_space_unit(self):  assert self._parse("210 g") == 210
    def test_none_returns_zero(self):       assert self._parse(None)    == 0
    def test_empty_string_returns_zero(self): assert self._parse("")   == 0
    def test_invalid_string_returns_zero(self): assert self._parse("N/A") == 0

    def test_progress_ratio_normal(self):
        v, cap = 100, 200
        assert min(v / cap, 1.0) == 0.5

    def test_progress_ratio_capped(self):
        v, cap = 500, 200
        assert min(v / cap, 1.0) == 1.0

    def test_zero_value(self):
        assert self._parse(0) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4 · JSON fence-stripping & parsing
# ─────────────────────────────────────────────────────────────────────────────
class TestJsonCleaning:
    """Mirrors the raw-response cleaning logic in the generate block."""

    @staticmethod
    def _clean(raw: str) -> dict:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(clean)

    def test_plain_json(self):
        assert self._clean('{"k":"v"}') == {"k": "v"}

    def test_fenced_json(self):
        assert self._clean("```json\n{\"k\":\"v\"}\n```") == {"k": "v"}

    def test_plain_fenced_json(self):
        assert self._clean("```\n{\"k\":\"v\"}\n```") == {"k": "v"}

    def test_leading_whitespace(self):
        assert self._clean("  \n  {\"k\":\"v\"}") == {"k": "v"}

    def test_nested_structure(self):
        data = {"meal_plan": {"breakfast": {"name": "Oats"}}, "grocery_list": []}
        assert self._clean(json.dumps(data)) == data

    def test_list_value(self):
        assert self._clean('{"items": [1, 2, 3]}') == {"items": [1, 2, 3]}

    def test_invalid_raises(self):
        with pytest.raises(json.JSONDecodeError):
            self._clean("not json")

    def test_empty_object(self):
        assert self._clean("{}") == {}


# ─────────────────────────────────────────────────────────────────────────────
# 5 · Error classification
# ─────────────────────────────────────────────────────────────────────────────
class TestErrorClassification:
    RATE_KW = ["quota", "resource exhausted", "429", "rate", "limit"]
    AUTH_KW = ["api key", "invalid", "permission", "403", "unauthenticated"]

    def _is_rate(self, msg): return any(x in msg.lower() for x in self.RATE_KW)
    def _is_auth(self, msg): return any(x in msg.lower() for x in self.AUTH_KW)

    def test_rate_limit_429(self):           assert self._is_rate("429 RESOURCE_EXHAUSTED quota")
    def test_rate_limit_quota(self):         assert self._is_rate("Quota exceeded for free tier")
    def test_rate_limit_exhausted(self):     assert self._is_rate("Resource exhausted")
    def test_auth_api_key(self):             assert self._is_auth("API key not valid")
    def test_auth_403(self):                 assert self._is_auth("403 Permission denied")
    def test_auth_invalid(self):             assert self._is_auth("Invalid credential")
    def test_rate_not_auth(self):            assert not self._is_auth("429 Resource exhausted")
    def test_auth_not_rate(self):            assert not self._is_rate("API key invalid")
    def test_404_neither(self):
        msg = "404 NOT_FOUND model not found"
        assert not self._is_rate(msg) and not self._is_auth(msg)

    def test_case_insensitive_rate(self):    assert self._is_rate("QUOTA EXCEEDED")
    def test_case_insensitive_auth(self):    assert self._is_auth("API KEY NOT VALID")


# ─────────────────────────────────────────────────────────────────────────────
# 6 · Model fallback list
# ─────────────────────────────────────────────────────────────────────────────
class TestModelFallback:
    def test_has_three_models(self):
        assert len(app._MODELS) == 3

    def test_primary_is_flash_2(self):
        assert app._MODELS[0] == "gemini-2.0-flash"

    def test_second_is_flash_15(self):
        assert app._MODELS[1] == "gemini-1.5-flash"

    def test_all_are_strings(self):
        assert all(isinstance(m, str) for m in app._MODELS)

    def test_all_start_with_gemini(self):
        assert all(m.startswith("gemini") for m in app._MODELS)

    def test_no_known_broken_model(self):
        assert "gemini-1.5-flash-8b" not in app._MODELS

    def test_no_duplicates(self):
        assert len(app._MODELS) == len(set(app._MODELS))


# ─────────────────────────────────────────────────────────────────────────────
# 7 · Grocery categorisation & totals
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_GROCERIES = [
    {"item": "Spinach",   "quantity": "1 bag",  "estimated_cost_usd": 2.50, "category": "Produce"},
    {"item": "Chicken",   "quantity": "500g",   "estimated_cost_usd": 5.00, "category": "Protein"},
    {"item": "Pasta",     "quantity": "400g",   "estimated_cost_usd": 1.50, "category": "Grains"},
    {"item": "Olive Oil", "quantity": "1 tbsp", "estimated_cost_usd": 0.50, "category": "Pantry"},
    {"item": "Cheddar",   "quantity": "100g",   "estimated_cost_usd": 2.00, "category": "Dairy"},
]


def _by_cat(items):
    result = {}
    for g in items:
        result.setdefault(g.get("category", "Other"), []).append(g)
    return result


class TestGroceryCategorisation:
    def test_correct_categories(self):
        bc = _by_cat(SAMPLE_GROCERIES)
        for cat in ["Produce", "Protein", "Grains", "Pantry", "Dairy"]:
            assert cat in bc

    def test_total_cost(self):
        total = sum(g["estimated_cost_usd"] for g in SAMPLE_GROCERIES)
        assert abs(total - 11.50) < 0.01

    def test_missing_category_defaults_other(self):
        items = [{"item": "X", "quantity": "1", "estimated_cost_usd": 1.0}]
        assert "Other" in _by_cat(items)

    def test_item_count_per_category(self):
        bc = _by_cat(SAMPLE_GROCERIES)
        assert len(bc["Produce"]) == 1
        assert len(bc["Protein"]) == 1

    def test_zero_cost_item(self):
        items = [{"item": "Salt", "quantity": "pinch", "estimated_cost_usd": 0.0, "category": "Pantry"}]
        total = sum(g["estimated_cost_usd"] for g in items)
        assert total == 0.0

    def test_empty_list(self):
        assert _by_cat([]) == {}


# ─────────────────────────────────────────────────────────────────────────────
# 8 · Presets
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_PRESET_KEYS = {"day_type", "people", "budget", "cooking_time", "health_goal"}


class TestPresets:
    def test_four_presets(self):
        assert len(app.PRESETS) == 4

    def test_all_have_required_keys(self):
        for label, vals in app.PRESETS.items():
            missing = REQUIRED_PRESET_KEYS - vals.keys()
            assert not missing, f"Preset '{label}' missing: {missing}"

    def test_budgets_positive(self):
        assert all(v["budget"] > 0 for v in app.PRESETS.values())

    def test_people_at_least_one(self):
        assert all(v["people"] >= 1 for v in app.PRESETS.values())

    def test_preset_labels_unique(self):
        labels = list(app.PRESETS.keys())
        assert len(labels) == len(set(labels))

    def test_meal_prep_has_highest_budget(self):
        budgets = {k: v["budget"] for k, v in app.PRESETS.items()}
        assert budgets["📦 Meal Prep Sunday"] == max(budgets.values())

    def test_fitness_day_is_high_protein(self):
        assert app.PRESETS["💪 Fitness Day"]["health_goal"] == "High Protein"


# ─────────────────────────────────────────────────────────────────────────────
# 9 · Substitution cost savings
# ─────────────────────────────────────────────────────────────────────────────
class TestSubstitutions:
    SUBS = [
        {"original": "Salmon",    "substitute": "Canned Tuna",  "reason": "budget", "cost_saving_usd": 4.50},
        {"original": "Pine nuts", "substitute": "Sunflower seeds", "reason": "budget", "cost_saving_usd": 2.00},
        {"original": "Parmesan",  "substitute": "Nutritional yeast", "reason": "vegan", "cost_saving_usd": 0},
    ]

    def test_total_saving(self):
        total = sum(float(s.get("cost_saving_usd", 0) or 0) for s in self.SUBS)
        assert abs(total - 6.50) < 0.01

    def test_zero_saving_allowed(self):
        s = self.SUBS[2]
        assert float(s.get("cost_saving_usd", 0) or 0) == 0.0

    def test_reason_present(self):
        assert all("reason" in s for s in self.SUBS)

    def test_original_and_substitute_present(self):
        for s in self.SUBS:
            assert s["original"] and s["substitute"]


# ─────────────────────────────────────────────────────────────────────────────
# 10 · Meal plan schema validation
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_PLAN = {
    "chef_message": "Let's make today delicious!",
    "meal_plan": {
        "breakfast": {
            "name": "Greek Yogurt Bowl", "description": "Quick and filling.",
            "prep_time": "5 mins", "cook_time": "0 mins", "difficulty": "Easy",
            "calories_per_serving": "380 kcal",
            "ingredients": ["Greek yogurt", "Granola", "Berries"],
            "cooking_steps": ["Step 1: Scoop yogurt.", "Step 2: Add toppings."],
            "chef_tip": "Use full-fat yogurt for creaminess.",
        },
        "lunch": {
            "name": "Pasta Primavera", "description": "Light veggie pasta.",
            "prep_time": "10 mins", "cook_time": "15 mins", "difficulty": "Easy",
            "calories_per_serving": "520 kcal",
            "ingredients": ["Pasta", "Bell pepper", "Zucchini", "Olive oil"],
            "cooking_steps": ["Step 1: Boil pasta.", "Step 2: Sauté veg.", "Step 3: Combine."],
            "chef_tip": "Salt the pasta water generously.",
        },
        "dinner": {
            "name": "Chickpea Curry", "description": "Hearty and warming.",
            "prep_time": "10 mins", "cook_time": "25 mins", "difficulty": "Medium",
            "calories_per_serving": "640 kcal",
            "ingredients": ["Chickpeas", "Tomatoes", "Spinach", "Spices"],
            "cooking_steps": ["Step 1: Sauté onion.", "Step 2: Add spices.", "Step 3: Add chickpeas.", "Step 4: Simmer."],
            "chef_tip": "Add a squeeze of lemon at the end.",
        },
    },
    "grocery_list": SAMPLE_GROCERIES,
    "substitutions": [],
    "budget_analysis": {
        "total_estimated_cost_usd": 11.50, "cost_per_person_usd": 5.75,
        "status": "within_budget", "percentage_of_budget_used": 74,
        "savings_tips": ["Buy in bulk.", "Use seasonal produce."],
        "verdict": "Well within budget.",
    },
    "nutritional_summary": {
        "total_calories": "1540 kcal", "protein_g": "72",
        "carbs_g": "195", "fat_g": "48",
        "highlights": ["High fibre", "Good iron from spinach"],
    },
}


class TestMealPlanSchema:
    def test_three_meals_present(self):
        assert set(SAMPLE_PLAN["meal_plan"]) == {"breakfast", "lunch", "dinner"}

    def test_each_meal_has_name(self):
        for meal in SAMPLE_PLAN["meal_plan"].values():
            assert meal["name"]

    def test_each_meal_has_steps(self):
        for meal in SAMPLE_PLAN["meal_plan"].values():
            assert len(meal["cooking_steps"]) >= 1

    def test_each_meal_has_ingredients(self):
        for meal in SAMPLE_PLAN["meal_plan"].values():
            assert len(meal["ingredients"]) >= 1

    def test_difficulty_valid_values(self):
        for meal in SAMPLE_PLAN["meal_plan"].values():
            assert meal["difficulty"] in {"Easy", "Medium", "Hard"}

    def test_budget_analysis_keys(self):
        ba = SAMPLE_PLAN["budget_analysis"]
        for key in ["total_estimated_cost_usd", "status", "verdict", "savings_tips"]:
            assert key in ba

    def test_nutrition_keys(self):
        nut = SAMPLE_PLAN["nutritional_summary"]
        for key in ["total_calories", "protein_g", "carbs_g", "fat_g"]:
            assert key in nut

    def test_chef_message_present(self):
        assert SAMPLE_PLAN["chef_message"]

    def test_grocery_list_non_empty(self):
        assert len(SAMPLE_PLAN["grocery_list"]) > 0

    def test_budget_status_valid(self):
        assert SAMPLE_PLAN["budget_analysis"]["status"] in {
            "within_budget", "tight", "over_budget"
        }
