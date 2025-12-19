from tools.spreadsheet import SpreadsheetTool


def test_try_answer_incident_count_handles_completed_and_june_2025():
    csv = (
        "Number,State,Caller,Update Date\n"
        "INC001,Completed,Garrette Griffin,2025-06-01\n"
        "INC002,Closed - Completed,Garrette L. Griffin,2025-06-15\n"
        "INC003,Completed,Someone Else,2025-06-20\n"
        "INC004,In Progress,Garrette Griffin,2025-06-10\n"
        "INC005,Completed,\"Griffin, Garrette\",2025-07-01\n"
    ).encode("utf-8")

    tool = SpreadsheetTool()
    tables, _profiles = tool.load_tables_from_bytes("incidents.csv", csv)

    q = "How many incidents did Garrette Griffin complete in June of 2025?"
    result = tool.try_answer_incident_count(q, tables)

    assert result is not None
    assert result["count"] == 2
