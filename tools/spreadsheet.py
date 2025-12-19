"""Spreadsheet Tool

Provides safe, computation-oriented access to uploaded spreadsheets (XLSX/XLS) and CSVs.

Design goals:
- No arbitrary code execution (SQL is SELECT-only)
- Avoid reading unbounded data into context; return compact profiles + query results
- Keep everything local
"""

from __future__ import annotations

import io
import calendar
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from config import settings


@dataclass
class TableProfile:
    table_name: str
    source_name: str
    rows: int
    cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    truncated: bool = False
    notes: Optional[str] = None


class SpreadsheetTool:
    """Spreadsheet tool backed by pandas for parsing and DuckDB for computation."""

    _DISALLOWED_SQL_TOKENS = (
        "attach",
        "copy",
        "create",
        "delete",
        "drop",
        "export",
        "import",
        "insert",
        "install",
        "load",
        "pragma",
        "replace",
        "update",
        "vacuum",
        "call",
        "alter",
    )

    def __init__(self):
        self.settings = settings

    @staticmethod
    def _sanitize_table_name(name: str) -> str:
        # DuckDB identifiers: keep alnum + underscore, lowercased.
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "").strip()).strip("_").lower()
        if not cleaned:
            cleaned = "table"
        if not re.match(r"^[a-zA-Z_]", cleaned):
            cleaned = f"t_{cleaned}"
        return cleaned

    @classmethod
    def is_safe_select_sql(cls, sql: str) -> Tuple[bool, str]:
        if not sql or not isinstance(sql, str):
            return False, "Empty SQL"

        normalized = sql.strip()
        if not normalized:
            return False, "Empty SQL"

        # Disallow multi-statements.
        if ";" in normalized:
            return False, "Only single-statement SELECT queries are allowed"

        lowered = normalized.lower()

        # Only SELECT or WITH (CTE) allowed.
        if not (lowered.startswith("select") or lowered.startswith("with")):
            return False, "Only SELECT queries are allowed"

        # Block known non-read operations.
        for token in cls._DISALLOWED_SQL_TOKENS:
            if re.search(rf"\b{re.escape(token)}\b", lowered):
                return False, f"Disallowed SQL token: {token}"

        return True, "ok"

    def _get_limits(self) -> Dict[str, int]:
        max_rows_per_table = int(getattr(self.settings, "spreadsheet_max_rows_per_table", 200_000))
        max_cells_per_table = int(getattr(self.settings, "spreadsheet_max_cells_per_table", 5_000_000))
        preview_rows = int(getattr(self.settings, "spreadsheet_preview_rows", 20))
        query_max_rows = int(getattr(self.settings, "spreadsheet_query_max_rows", 200))

        # Defensive lower bounds
        preview_rows = max(5, preview_rows)
        query_max_rows = max(50, query_max_rows)
        max_rows_per_table = max(1_000, max_rows_per_table)
        max_cells_per_table = max(100_000, max_cells_per_table)

        return {
            "max_rows_per_table": max_rows_per_table,
            "max_cells_per_table": max_cells_per_table,
            "preview_rows": preview_rows,
            "query_max_rows": query_max_rows,
        }

    def _import_pandas(self):
        try:
            import pandas as pd  # type: ignore

            return pd
        except Exception as e:
            raise RuntimeError(
                "Spreadsheet feature requires pandas. Install dependencies from requirements.txt"
            ) from e

    def _import_duckdb(self):
        try:
            import duckdb  # type: ignore

            return duckdb
        except Exception as e:
            raise RuntimeError(
                "Spreadsheet computation requires duckdb. Add it to requirements.txt and install."
            ) from e

    def load_tables_from_bytes(self, filename: str, file_bytes: bytes) -> Tuple[Dict[str, Any], List[TableProfile]]:
        """Load a workbook/CSV into a dict of {table_name: DataFrame}.

        Returns:
            (tables, profiles)
        """
        pd = self._import_pandas()
        limits = self._get_limits()

        ext = Path(filename).suffix.lower()
        tables: Dict[str, Any] = {}
        profiles: List[TableProfile] = []

        if ext in {".xlsx", ".xls"}:
            # Read all sheets.
            # NOTE: pandas uses openpyxl for xlsx; xls may require optional engines.
            with io.BytesIO(file_bytes) as bio:
                try:
                    sheet_map: Mapping[str, Any] = pd.read_excel(bio, sheet_name=None, engine=None)
                except Exception:
                    # Try explicit engine for xlsx
                    bio.seek(0)
                    sheet_map = pd.read_excel(bio, sheet_name=None, engine="openpyxl")

            for sheet_name, df in sheet_map.items():
                table_name = self._sanitize_table_name(f"{Path(filename).stem}_{sheet_name}")
                df2, truncated, note = self._apply_limits(df, limits)
                tables[table_name] = df2
                profiles.append(self._profile_table(table_name, f"{filename}:{sheet_name}", df2, truncated, note))

        elif ext == ".csv":
            with io.BytesIO(file_bytes) as bio:
                df = pd.read_csv(bio)
            table_name = self._sanitize_table_name(Path(filename).stem)
            df2, truncated, note = self._apply_limits(df, limits)
            tables[table_name] = df2
            profiles.append(self._profile_table(table_name, filename, df2, truncated, note))
        else:
            raise ValueError(f"Unsupported structured file type: {ext}")

        return tables, profiles

    def _apply_limits(self, df: Any, limits: Dict[str, int]) -> Tuple[Any, bool, Optional[str]]:
        """Apply row/cell limits to bound memory usage."""
        rows = int(getattr(df, "shape", (0, 0))[0])
        cols = int(getattr(df, "shape", (0, 0))[1])
        truncated = False
        note: Optional[str] = None

        max_rows = limits["max_rows_per_table"]
        max_cells = limits["max_cells_per_table"]

        # Truncate by rows first
        if rows > max_rows:
            df = df.head(max_rows)
            truncated = True
            note = f"Truncated to first {max_rows:,} rows"
            rows = max_rows

        # Then bound total cells by further trimming rows
        if cols > 0 and rows * cols > max_cells:
            allowed_rows = max(1, max_cells // cols)
            if allowed_rows < rows:
                df = df.head(allowed_rows)
                truncated = True
                note = f"Truncated to first {allowed_rows:,} rows to cap {max_cells:,} cells"

        return df, truncated, note

    def _profile_table(
        self,
        table_name: str,
        source_name: str,
        df: Any,
        truncated: bool,
        note: Optional[str],
    ) -> TableProfile:
        dtypes: Dict[str, str] = {}
        try:
            # pandas DataFrame
            for col, dtype in getattr(df, "dtypes", {}).items():
                dtypes[str(col)] = str(dtype)
        except Exception:
            dtypes = {}

        rows = int(getattr(df, "shape", (0, 0))[0])
        cols = int(getattr(df, "shape", (0, 0))[1])
        columns = [str(c) for c in list(getattr(df, "columns", []))]

        return TableProfile(
            table_name=table_name,
            source_name=source_name,
            rows=rows,
            cols=cols,
            columns=columns,
            dtypes=dtypes,
            truncated=truncated,
            notes=note,
        )

    def format_profiles(self, profiles: List[TableProfile]) -> str:
        lines: List[str] = []
        lines.append("[Structured Data Tables]")
        for p in profiles:
            trunc = " (TRUNCATED)" if p.truncated else ""
            lines.append(f"- {p.table_name} <- {p.source_name}{trunc}")
            lines.append(f"  rows={p.rows:,} cols={p.cols:,}")
            if p.columns:
                lines.append("  columns: " + ", ".join(p.columns[:60]) + (" …" if len(p.columns) > 60 else ""))
            if p.notes:
                lines.append(f"  note: {p.notes}")
        return "\n".join(lines)

    def format_table_insights(self, tables: Dict[str, Any], profiles: List[TableProfile]) -> str:
        """Deterministic, value-level profiling for tables.

        Goal: give the model concrete evidence (top values, date ranges, etc.) so it
        can reliably answer "easy" spreadsheet questions without guessing.
        """

        pd = self._import_pandas()
        limits = self._get_limits()

        max_cols = int(getattr(self.settings, "spreadsheet_profile_max_cols", 30) or 30)
        max_rows_sample = int(getattr(self.settings, "spreadsheet_profile_sample_rows", min(10, limits["preview_rows"])) or 10)
        top_n = int(getattr(self.settings, "spreadsheet_profile_top_n", 5) or 5)
        max_text_len = int(getattr(self.settings, "spreadsheet_profile_max_text_len", 9000) or 9000)

        lines: List[str] = ["[Spreadsheet Profile]"]

        for p in profiles:
            df = (tables or {}).get(p.table_name)
            if df is None:
                continue

            try:
                cols = [str(c) for c in getattr(df, "columns", [])]
            except Exception:
                cols = []

            lines.append(f"- {p.table_name} <- {p.source_name} (rows={p.rows:,} cols={p.cols:,})")

            # Highlight likely key columns (common in operational exports).
            detected = self._detect_common_columns(df, cols)
            if detected:
                pretty = ", ".join(f"{k}='{v}'" for k, v in detected.items() if v)
                if pretty:
                    lines.append(f"  likely columns: {pretty}")

            # Column-by-column profile (bounded).
            for col in cols[:max_cols]:
                try:
                    s = df[col]
                except Exception:
                    continue

                dtype = str(getattr(s, "dtype", ""))
                try:
                    nulls = int(s.isna().sum())
                    total = int(len(s))
                except Exception:
                    nulls = 0
                    total = 0

                non_null = max(0, total - nulls)
                try:
                    unique = int(s.nunique(dropna=True))
                except Exception:
                    unique = 0

                sem = self._infer_semantic_kind(pd, s, col_name=col, non_null=non_null)
                sem_text = ""
                if sem and sem.get("kind"):
                    conf = sem.get("confidence")
                    sem_text = f"semantic={sem.get('kind')}" + (f"({conf:.0%})" if isinstance(conf, float) else "")

                detail = self._summarize_series(pd, s, dtype=dtype, non_null=non_null, top_n=top_n)
                examples = self._example_values(pd, s, unique=unique, non_null=non_null, k=3)
                extra = "; ".join([x for x in [sem_text, detail, (f"examples={examples}" if examples else "")] if x])
                lines.append(
                    f"  - {col} ({dtype}) nulls={nulls:,}/{total:,} unique={unique:,}" + (f"; {extra}" if extra else "")
                )

            # A small row sample (bounded) to expose relationships between columns.
            try:
                sample_df = df.head(max_rows_sample)
                rows = sample_df.to_dict(orient="records")
                if rows:
                    lines.append("  sample_rows:")
                    for r in rows[:max_rows_sample]:
                        # Keep each row compact.
                        parts = []
                        for k, v in list(r.items())[:max_cols]:
                            parts.append(f"{k}={self._truncate(str(v), 80)}")
                        lines.append("    - " + ", ".join(parts))
            except Exception:
                pass

            # Hard cap overall profile size.
            if sum(len(x) for x in lines) > max_text_len:
                lines.append("[Spreadsheet Profile truncated]")
                break

        return "\n".join(lines)

    def build_planner_hints(self, tables: Dict[str, Any], profiles: List[TableProfile]) -> str:
        """Compact hints intended specifically for the SQL planner call."""

        text = self.format_table_insights(tables, profiles)
        # Keep planner prompt small; the full profile is already in chat context.
        return self._truncate(text, 6000)

    def _detect_common_columns(self, df: Any, columns: List[str]) -> Dict[str, Optional[str]]:
        """Detect common semantic columns using both headers and values.

        Enterprises don't rely on LLM guesses for schema understanding. They do
        deterministic profiling + type/semantic inference, then run computation.
        """

        # Header-based candidates.
        header_status = self._pick_best_column(columns, ["state", "status", "stage", "resolution", "close", "completed", "complete"])
        header_person = self._pick_best_column(
            columns,
            [
                "caller",
                "requested",
                "requester",
                "opened_by",
                "opened by",
                "created by",
                "assignee",
                "assigned",
                "user",
                "employee",
                "name",
                "submitted by",
                "reported by",
                "owner",
            ],
        )
        header_date = self._pick_best_column(
            columns,
            [
                "update",
                "updated",
                "last update",
                "last_updated",
                "closed",
                "resolved",
                "resolution",
                "date",
                "time",
                "timestamp",
                "created",
                "opened",
            ],
        )
        header_id = self._pick_best_column(columns, ["id", "number", "incident", "ticket", "case", "ref", "reference"])

        # Value-based inference.
        pd = self._import_pandas()
        best: Dict[str, Tuple[Optional[str], float]] = {
            "status": (header_status, 0.0),
            "person": (header_person, 0.0),
            "date": (header_date, 0.0),
            "id": (header_id, 0.0),
        }

        for col in columns:
            try:
                s = df[col]
                non_null = int(s.notna().sum())
            except Exception:
                continue

            sem = self._infer_semantic_kind(pd, s, col_name=col, non_null=non_null)
            if not sem:
                continue

            kind = sem.get("kind")
            conf = float(sem.get("confidence") or 0.0)
            if kind in best and conf > best[kind][1]:
                best[kind] = (col, conf)

        return {
            "status": best["status"][0],
            "person": best["person"][0],
            "date": best["date"][0],
            "id": best["id"][0],
        }

    @classmethod
    def _summarize_series(cls, pd, s: Any, dtype: str, non_null: int, top_n: int) -> str:
        """Return a short, value-level summary for a pandas Series."""

        if non_null <= 0:
            return ""

        # Try datetime.
        try:
            dt = pd.to_datetime(s, errors="coerce")
            dt_non_null = int(dt.notna().sum())
        except Exception:
            dt = None
            dt_non_null = 0

        # Treat as date if most non-null values parse as datetimes.
        if dt is not None and dt_non_null >= max(3, int(non_null * 0.7)):
            try:
                dmin = dt.min()
                dmax = dt.max()
                return f"date_range={str(dmin)[:10]}..{str(dmax)[:10]}"
            except Exception:
                return ""

        # Numeric stats.
        try:
            if pd.api.types.is_numeric_dtype(s):
                s_num = pd.to_numeric(s, errors="coerce")
                s_num = s_num.dropna()
                if len(s_num) > 0:
                    return f"min={s_num.min():g} max={s_num.max():g} mean={s_num.mean():g}"
        except Exception:
            pass

        # Categorical/text: show top values.
        try:
            vc = s.astype(str).str.strip().replace({"nan": ""})
            vc = vc[vc != ""]
            counts = vc.value_counts(dropna=True).head(top_n)
            if len(counts) == 0:
                return ""
            items = [f"{cls._truncate(str(k), 30)}({int(v)})" for k, v in counts.items()]
            return "top=" + ", ".join(items)
        except Exception:
            return ""

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if text is None:
            return ""
        t = str(text)
        if len(t) <= max_len:
            return t
        return t[: max(0, max_len - 1)] + "…"

    @classmethod
    def _example_values(cls, pd, s: Any, unique: int, non_null: int, k: int) -> str:
        """Provide a few example values when a column is high-cardinality.

        This helps with columns like names where value_counts(top) can be
        misleading because most values appear once.
        """
        if non_null <= 0:
            return ""
        if unique <= 0 or unique < max(20, int(non_null * 0.6)):
            return ""
        try:
            vals = s.dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            uniq: List[str] = []
            seen = set()
            for v in vals.head(300).tolist():
                v2 = " ".join(v.split())
                if not v2 or v2 in seen:
                    continue
                seen.add(v2)
                uniq.append(cls._truncate(v2, 30))
                if len(uniq) >= k:
                    break
            return ", ".join(uniq)
        except Exception:
            return ""

    @classmethod
    def _infer_semantic_kind(cls, pd, s: Any, col_name: str, non_null: int) -> Dict[str, Any]:
        """Infer a semantic kind for a column using header + value heuristics."""

        if non_null <= 0:
            return {}

        header_scores = {
            "person": cls._score_column(
                col_name,
                [
                    "caller",
                    "requester",
                    "requested",
                    "opened by",
                    "created by",
                    "assignee",
                    "user",
                    "employee",
                    "name",
                    "owner",
                    "reported by",
                    "submitted by",
                ],
            ),
            "status": cls._score_column(col_name, ["state", "status", "stage", "resolution", "close", "completed", "complete"]),
            "date": cls._score_column(col_name, ["date", "time", "timestamp", "updated", "update", "created", "opened", "closed", "resolved"]),
            "id": cls._score_column(col_name, ["id", "number", "incident", "ticket", "case", "ref", "reference"]),
        }

        # Sample values deterministically.
        try:
            vals = s.dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            sample = vals.head(200).tolist()
        except Exception:
            sample = []

        if not sample:
            best_kind = max(header_scores, key=lambda k: header_scores[k])
            if header_scores[best_kind] >= 10:
                return {"kind": best_kind, "confidence": min(1.0, header_scores[best_kind] / 50.0)}
            return {}

        person_like = cls._fraction_person_like(sample)
        email_like = cls._fraction_email_like(sample)
        status_like = cls._fraction_status_like(sample)
        id_like = cls._fraction_id_like(sample)

        try:
            dt = pd.to_datetime(pd.Series(sample), errors="coerce")
            date_like = float(dt.notna().mean())
        except Exception:
            date_like = 0.0

        scores = {
            "person": (0.15 * min(1.0, header_scores["person"] / 50.0)) + (0.60 * max(person_like, email_like)),
            "status": (0.20 * min(1.0, header_scores["status"] / 50.0)) + (0.60 * status_like),
            "date": (0.20 * min(1.0, header_scores["date"] / 50.0)) + (0.65 * date_like),
            "id": (0.20 * min(1.0, header_scores["id"] / 50.0)) + (0.55 * id_like),
        }

        best_kind = max(scores, key=lambda k: scores[k])
        best_score = float(scores[best_kind])
        if best_score < 0.35:
            return {}
        return {"kind": best_kind, "confidence": min(1.0, best_score)}

    @staticmethod
    def _fraction_email_like(values: List[str]) -> float:
        if not values:
            return 0.0
        re_email = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
        hits = 0
        for v in values:
            if re_email.match(v.strip().lower()):
                hits += 1
        return hits / max(1, len(values))

    @classmethod
    def _fraction_person_like(cls, values: List[str]) -> float:
        if not values:
            return 0.0
        hits = 0
        for raw in values:
            v = " ".join((raw or "").strip().split())
            if not v:
                continue
            if any(ch.isdigit() for ch in v):
                continue
            if "@" in v:
                continue
            tokens = cls._tokenize(v)
            if len(tokens) < 2 or len(tokens) > 4:
                continue
            if not all(t.isalpha() for t in tokens):
                continue
            if len(v) < 5 or len(v) > 50:
                continue
            hits += 1
        return hits / max(1, len(values))

    @staticmethod
    def _fraction_status_like(values: List[str]) -> float:
        if not values:
            return 0.0
        keywords = ["complete", "completed", "closed", "resolved", "open", "new", "in progress", "pending", "cancel"]
        hits = 0
        for v in values:
            vl = v.strip().lower()
            if any(k in vl for k in keywords):
                hits += 1
        return hits / max(1, len(values))

    @staticmethod
    def _fraction_id_like(values: List[str]) -> float:
        if not values:
            return 0.0
        re_ticket = re.compile(r"^[A-Za-z]{1,6}[-_ ]?\d{2,}$")
        hits = 0
        for v in values:
            if re_ticket.match(v.strip()):
                hits += 1
        return hits / max(1, len(values))

    def run_query(self, tables: Dict[str, Any], sql: str) -> Dict[str, Any]:
        """Execute a safe SELECT query over loaded tables."""
        ok, reason = self.is_safe_select_sql(sql)
        if not ok:
            raise ValueError(reason)

        duckdb = self._import_duckdb()
        limits = self._get_limits()
        max_rows = limits["query_max_rows"]

        con = duckdb.connect(database=":memory:")
        try:
            for name, df in tables.items():
                con.register(name, df)

            # Enforce LIMIT by wrapping the query.
            wrapped = f"SELECT * FROM ({sql}) AS q LIMIT {max_rows}"
            try:
                result_df = con.execute(wrapped).df()
            except Exception as e:
                # Common failure mode: spreadsheets often have column headers with spaces
                # (e.g., "Sales Amt") and model-generated SQL forgets to quote them.
                # We do a best-effort retry by quoting any "non-identifier" columns.
                msg = str(e)
                if "Parser Error" not in msg:
                    raise

                fixed_sql = self._auto_quote_problem_identifiers(sql, tables)
                if fixed_sql == sql:
                    raise
                wrapped_fixed = f"SELECT * FROM ({fixed_sql}) AS q LIMIT {max_rows}"
                result_df = con.execute(wrapped_fixed).df()

            # Convert to compact structure.
            rows: List[Dict[str, Any]] = result_df.to_dict(orient="records")
            return {
                "sql": sql,
                "row_count": len(rows),
                "max_rows": max_rows,
                "columns": list(result_df.columns),
                "rows": rows,
            }
        finally:
            try:
                con.close()
            except Exception:
                pass

    def try_answer_incident_count(self, question: str, tables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Best-effort deterministic answer for simple incident count questions.

        This exists because LLM-authored SQL is brittle for "easy" questions when it
        selects the wrong column names or uses overly strict equality.

        Pattern supported (loosely):
        - "How many incidents did <person> complete in <month> <year>?"

        Returns:
            dict with keys: count, person, month, year, table, columns_used
            or None if the question doesn't match / cannot be answered safely.
        """

        parsed = self._parse_incident_count_question(question)
        if not parsed:
            return None

        pd = self._import_pandas()

        person = parsed["person"]
        month = parsed["month"]
        year = parsed["year"]

        start = pd.Timestamp(year=year, month=month, day=1)
        end = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])

        best: Optional[Dict[str, Any]] = None
        for table_name, df in (tables or {}).items():
            try:
                cols = [str(c) for c in getattr(df, "columns", [])]
            except Exception:
                continue

            status_col = self._pick_best_column(cols, ["state", "status", "resolution", "close", "completed", "complete"])
            person_col = self._pick_best_column(cols, ["caller", "requested", "requester", "opened_by", "opened by", "created by", "assignee", "assigned"])
            date_col = self._pick_best_column(cols, ["update", "updated", "last update", "last_updated", "closed", "resolved", "resolution", "date"])

            if not (status_col and person_col and date_col):
                continue

            try:
                s_status = df[status_col]
                s_person = df[person_col]
                s_date = df[date_col]
            except Exception:
                continue

            # Normalize values.
            status_norm = s_status.astype(str).str.strip().str.lower()
            person_norm = s_person.astype(str).str.strip().str.lower()

            # Date parsing: handles datetime, ISO strings, and common US formats.
            dt = pd.to_datetime(s_date, errors="coerce")

            # Build person match variants and token containment.
            person_variants = self._person_variants(person)
            person_tokens = set(self._tokenize(person))

            def person_match_one(v: str) -> bool:
                v_tokens = set(self._tokenize(v))
                if not v_tokens:
                    return False
                if v in person_variants:
                    return True
                # Conservative: require all person tokens present in the value.
                return bool(person_tokens) and person_tokens.issubset(v_tokens)

            # Person match: exact normalized match OR token containment.
            # Even if exact matches exist in the data, still include token-subset matches
            # (e.g., "Garrette L. Griffin" should match "Garrette Griffin").
            person_mask = person_norm.isin(person_variants) | person_norm.apply(person_match_one)

            # Status match: treat "completed" as substring to handle values like "Closed - Completed".
            status_mask = status_norm.str.contains("complete", na=False)

            # Date match within month.
            date_mask = (dt >= start) & (dt <= end)

            mask = person_mask & status_mask & date_mask
            try:
                count = int(mask.sum())
            except Exception:
                continue

            candidate = {
                "count": count,
                "person": person,
                "month": month,
                "year": year,
                "table": table_name,
                "columns_used": {
                    "status": status_col,
                    "person": person_col,
                    "date": date_col,
                },
            }

            # Prefer non-zero results; otherwise keep the best-scoring column picks.
            if best is None:
                best = candidate
            elif best.get("count", 0) == 0 and count > 0:
                best = candidate

        if not best:
            return None

        # If we only found zeros everywhere, it might still be correct. Return best.
        return best

    @staticmethod
    def _parse_incident_count_question(question: str) -> Optional[Dict[str, Any]]:
        if not question:
            return None

        q = " ".join(str(question).strip().split())
        ql = q.lower()
        if "incident" not in ql:
            return None
        if "how many" not in ql and "count" not in ql:
            return None
        if "complete" not in ql:
            return None

        # Capture: did <person> ... in <month> <year>
        m = re.search(
            r"how many\s+incidents\s+did\s+(?P<person>.+?)\s+(?:complete|completed)\s+(?:in|during)\s+(?P<month>[A-Za-z]+)\s+(?:of\s+)?(?P<year>\d{4})",
            ql,
            flags=re.IGNORECASE,
        )
        if not m:
            return None

        person_raw = m.group("person")
        person = " ".join(person_raw.strip().split())
        month_name = m.group("month").strip()
        year = int(m.group("year"))

        month = SpreadsheetTool._month_name_to_number(month_name)
        if not month:
            return None

        return {"person": person, "month": month, "year": year}

    @staticmethod
    def _month_name_to_number(month: str) -> Optional[int]:
        if not month:
            return None
        m = month.strip().lower()
        months = {
            "january": 1,
            "jan": 1,
            "february": 2,
            "feb": 2,
            "march": 3,
            "mar": 3,
            "april": 4,
            "apr": 4,
            "may": 5,
            "june": 6,
            "jun": 6,
            "july": 7,
            "jul": 7,
            "august": 8,
            "aug": 8,
            "september": 9,
            "sep": 9,
            "sept": 9,
            "october": 10,
            "oct": 10,
            "november": 11,
            "nov": 11,
            "december": 12,
            "dec": 12,
        }
        return months.get(m)

    @staticmethod
    def _tokenize(value: str) -> List[str]:
        v = (value or "").lower()
        return [t for t in re.split(r"[^a-z0-9]+", v) if t]

    @classmethod
    def _person_variants(cls, person: str) -> set:
        p = " ".join((person or "").strip().split()).lower()
        if not p:
            return set()
        tokens = cls._tokenize(p)
        variants = {p}
        if len(tokens) >= 2:
            first = tokens[0]
            last = tokens[-1]
            variants.add(f"{first} {last}")
            variants.add(f"{last}, {first}")
            variants.add(f"{last} {first}")
        return {" ".join(v.strip().split()) for v in variants if v.strip()}

    @classmethod
    def _pick_best_column(cls, columns: List[str], keywords: List[str]) -> Optional[str]:
        best_col = None
        best_score = 0
        for c in columns:
            score = cls._score_column(c, keywords)
            if score > best_score:
                best_score = score
                best_col = c
        return best_col

    @staticmethod
    def _score_column(column_name: str, keywords: List[str]) -> int:
        if not column_name:
            return 0
        name = column_name.strip().lower()
        name = re.sub(r"\s+", " ", name)
        score = 0
        for kw in keywords:
            k = kw.strip().lower()
            if not k:
                continue
            if k == name:
                score += 50
            elif k in name:
                score += 10
        # Prefer shorter / more specific column names a bit.
        if score and len(name) <= 12:
            score += 2
        return score

    @staticmethod
    def _auto_quote_problem_identifiers(sql: str, tables: Dict[str, Any]) -> str:
        """Best-effort fix for unquoted spreadsheet column headers.

        DuckDB (and SQL in general) requires quoting identifiers that contain spaces
        or punctuation. Model-generated SQL frequently emits `Sales Amt` instead of
        `"Sales Amt"`, causing a Parser Error.

        This function retries by quoting any column name that is not already a simple
        SQL identifier (letters/numbers/underscore) and appears in the SQL.
        """
        if not sql:
            return sql

        # Collect candidate column names.
        candidates: List[str] = []
        for df in (tables or {}).values():
            try:
                cols = list(getattr(df, "columns", []))
            except Exception:
                cols = []
            for col in cols:
                col_s = str(col)
                # Only quote columns that require it (spaces/punctuation/etc).
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col_s):
                    candidates.append(col_s)

        if not candidates:
            return sql

        fixed = sql
        # Replace longer names first to avoid partial overlap.
        for col in sorted(set(candidates), key=len, reverse=True):
            escaped = re.escape(col)
            # Avoid double-quoting if the identifier is already quoted.
            fixed = re.sub(rf'(?<!")({escaped})(?!")', r'"\1"', fixed)

        return fixed

    def format_query_result(self, result: Dict[str, Any]) -> str:
        """Format query result for LLM context (compact, readable)."""
        cols = result.get("columns") or []
        rows = result.get("rows") or []
        max_rows = result.get("max_rows")

        # Simple markdown-ish table to keep it readable.
        lines: List[str] = []
        lines.append("[Spreadsheet Query Result]")
        if cols:
            lines.append(" | ".join(str(c) for c in cols))
            lines.append(" | ".join(["---"] * len(cols)))
            for r in rows:
                lines.append(" | ".join(str(r.get(c, "")) for c in cols))
        else:
            lines.append("(no rows)")

        if len(rows) >= int(max_rows or 0):
            lines.append(f"(showing first {max_rows} rows)")

        return "\n".join(lines)
