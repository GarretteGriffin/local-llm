from tools.spreadsheet import SpreadsheetTool


def test_format_table_insights_includes_top_values_and_date_range():
    csv = (
        "Number,State,Caller,Update Date,Amount\n"
        "INC001,Completed,Garrette Griffin,2025-06-01,10\n"
        "INC002,Closed - Completed,Garrette L. Griffin,2025-06-15,20\n"
        "INC003,Completed,Someone Else,2025-06-20,30\n"
        "INC004,In Progress,Garrette Griffin,2025-06-10,5\n"
    ).encode("utf-8")

    tool = SpreadsheetTool()
    tables, profiles = tool.load_tables_from_bytes("incidents.csv", csv)

    text = tool.format_table_insights(tables, profiles)

    assert "[Spreadsheet Profile]" in text
    # Should surface categorical top values.
    assert "Completed" in text
    # Should surface a date range summary.
    assert "date_range=2025-06-01..2025-06-20" in text
