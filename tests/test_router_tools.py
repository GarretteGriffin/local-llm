from core.router import QueryRouter
from config import ToolType, ModelTier


def test_router_adds_spreadsheet_tool_for_structured_files():
    r = QueryRouter()
    d = r.route("summarize this", has_files=True, file_types=[".xlsx"], has_images=False)
    assert ToolType.SPREADSHEET in d.tools
    assert d.tier == ModelTier.POWER


def test_router_does_not_use_quick_tier_with_files_even_if_how_many():
    r = QueryRouter()
    d = r.route(
        "How many incidents did Garrette Griffin complete in June of 2025?",
        has_files=True,
        file_types=[".xlsx"],
        has_images=False,
    )
    assert ToolType.SPREADSHEET in d.tools
    assert d.tier == ModelTier.POWER


def test_router_adds_calculator_tool_for_math():
    r = QueryRouter()
    d = r.route("(2+3)*4", has_files=False, file_types=None, has_images=False)
    assert ToolType.CALCULATOR in d.tools


def test_router_routes_planning_to_power_more_often():
    r = QueryRouter()
    d = r.route("Plan and implement a solution to deduplicate customers and compute churn.", has_files=False)
    assert d.tier in (ModelTier.POWER, ModelTier.STANDARD)
