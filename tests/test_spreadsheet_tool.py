import pytest

from tools.spreadsheet import SpreadsheetTool


def test_sanitize_table_name_is_stable():
    tool = SpreadsheetTool()
    assert tool._sanitize_table_name("My Sheet") == "my_sheet"
    assert tool._sanitize_table_name("123") == "t_123"
    assert tool._sanitize_table_name("  ") == "table"
    assert tool._sanitize_table_name("A/B/C") == "a_b_c"


def test_sql_safety_guard():
    ok, _ = SpreadsheetTool.is_safe_select_sql("select 1")
    assert ok

    ok, reason = SpreadsheetTool.is_safe_select_sql("insert into t values (1)")
    assert not ok
    assert "Only SELECT" in reason or "Disallowed" in reason

    ok, reason = SpreadsheetTool.is_safe_select_sql("select 1; select 2")
    assert not ok
    assert "single-statement" in reason.lower()

    ok, reason = SpreadsheetTool.is_safe_select_sql("pragma show_tables")
    assert not ok


def test_run_query_simple_aggregate():
    duckdb = pytest.importorskip("duckdb")

    import pandas as pd

    tool = SpreadsheetTool()
    df = pd.DataFrame({"dept": ["A", "A", "B"], "amount": [10, 20, 7]})
    result = tool.run_query({"sales": df}, "select dept, sum(amount) as total from sales group by dept order by dept")

    assert result["columns"] == ["dept", "total"]
    assert result["row_count"] == 2
    assert result["rows"][0]["dept"] == "A"
    assert int(result["rows"][0]["total"]) == 30


def test_run_query_auto_quotes_space_columns():
    pytest.importorskip("duckdb")

    import pandas as pd

    tool = SpreadsheetTool()
    df = pd.DataFrame({"Division": ["D1", "D1", "D2"], "Sales Amt": [10, 20, 7]})

    # Intentionally unquoted identifier with a space; DuckDB would normally throw a Parser Error.
    result = tool.run_query(
        {"sales_data": df},
        "select Division, sum(Sales Amt) as TotalSales from sales_data group by Division order by Division",
    )

    assert result["columns"] == ["Division", "TotalSales"]
    assert result["row_count"] == 2
