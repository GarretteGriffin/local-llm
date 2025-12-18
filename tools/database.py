"""
Database Tool - Handles database files and QVD files.
Supports: SQLite, Access (mdb/accdb), QVD (QlikView)
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import tempfile

from config import settings


@dataclass
class DatabaseContent:
    """Extracted database content"""
    filename: str
    db_type: str
    tables: List[str]
    content: str
    row_count: int = 0
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatabaseTool:
    """
    Reads and queries database files.
    Automatically extracts schema and sample data for LLM context.
    """
    
    SUPPORTED_TYPES = {
        '.db': 'sqlite',
        '.sqlite': 'sqlite',
        '.sqlite3': 'sqlite',
        '.mdb': 'access',
        '.accdb': 'access',
        '.qvd': 'qvd',
    }
    
    def __init__(self):
        self.settings = settings
        self.max_rows = 100  # Sample rows per table
    
    def read(self, file_path: Union[str, Path], file_bytes: Optional[bytes] = None) -> DatabaseContent:
        """
        Read database file and extract content.
        
        Args:
            file_path: Path to database file
            file_bytes: Optional file bytes for uploads
            
        Returns:
            DatabaseContent with schema and sample data
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        filename = path.name
        
        if extension not in self.SUPPORTED_TYPES:
            return DatabaseContent(
                filename=filename,
                db_type='unknown',
                tables=[],
                content="",
                error=f"Unsupported database type: {extension}"
            )
        
        db_type = self.SUPPORTED_TYPES[extension]
        
        # Handle uploaded bytes
        if file_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name
            try:
                result = self._read_by_type(temp_path, db_type, filename)
            finally:
                os.unlink(temp_path)
            return result
        
        if not path.exists():
            return DatabaseContent(
                filename=filename,
                db_type=db_type,
                tables=[],
                content="",
                error=f"File not found: {file_path}"
            )
        
        return self._read_by_type(str(path), db_type, filename)
    
    def _read_by_type(self, file_path: str, db_type: str, filename: str) -> DatabaseContent:
        """Read database based on type"""
        try:
            if db_type == 'sqlite':
                return self._read_sqlite(file_path, filename)
            elif db_type == 'access':
                return self._read_access(file_path, filename)
            elif db_type == 'qvd':
                return self._read_qvd(file_path, filename)
            else:
                return DatabaseContent(
                    filename=filename,
                    db_type=db_type,
                    tables=[],
                    content="",
                    error=f"Reader not implemented: {db_type}"
                )
        except Exception as e:
            return DatabaseContent(
                filename=filename,
                db_type=db_type,
                tables=[],
                content="",
                error=f"Error reading database: {str(e)}"
            )
    
    def _read_sqlite(self, file_path: str, filename: str) -> DatabaseContent:
        """Read SQLite database"""
        import sqlite3

        content_parts = []
        total_rows = 0

        with sqlite3.connect(file_path) as conn:
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                # Get schema
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                col_names = [col[1] for col in columns]
                col_types = [col[2] for col in columns]

                content_parts.append(f"\n[Table: {table}]")
                content_parts.append(f"Columns: {', '.join(f'{n} ({t})' for n, t in zip(col_names, col_types))}")

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                total_rows += count
                content_parts.append(f"Rows: {count}")

                # Get sample data
                cursor.execute(f"SELECT * FROM {table} LIMIT {self.max_rows}")
                rows = cursor.fetchall()

                if rows:
                    content_parts.append("Sample data:")
                    content_parts.append(" | ".join(col_names))
                    content_parts.append("-" * 40)
                    for row in rows[:10]:  # Show first 10
                        content_parts.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
                    if len(rows) > 10:
                        content_parts.append(f"... ({len(rows)} rows shown)")
        
        return DatabaseContent(
            filename=filename,
            db_type='sqlite',
            tables=tables,
            content="\n".join(content_parts),
            row_count=total_rows,
            metadata={'table_count': len(tables)}
        )
    
    def _read_access(self, file_path: str, filename: str) -> DatabaseContent:
        """Read Microsoft Access database"""
        try:
            import pyodbc
        except ImportError:
            return DatabaseContent(
                filename=filename,
                db_type='access',
                tables=[],
                content="",
                error="pyodbc not installed. Install with: pip install pyodbc"
            )
        
        # Connection string for Access
        conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            f'DBQ={file_path};'
        )
        
        conn = None
        try:
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            
            # Get tables
            tables = [row.table_name for row in cursor.tables(tableType='TABLE')]
            
            content_parts = []
            total_rows = 0
            
            for table in tables:
                content_parts.append(f"\n[Table: {table}]")
                
                # Get columns
                cursor.execute(f"SELECT TOP 1 * FROM [{table}]")
                columns = [desc[0] for desc in cursor.description]
                content_parts.append(f"Columns: {', '.join(columns)}")
                
                # Get count and sample
                cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                count = cursor.fetchone()[0]
                total_rows += count
                content_parts.append(f"Rows: {count}")
                
                cursor.execute(f"SELECT TOP {self.max_rows} * FROM [{table}]")
                rows = cursor.fetchall()
                
                if rows:
                    content_parts.append("Sample data:")
                    content_parts.append(" | ".join(columns))
                    for row in rows[:10]:
                        content_parts.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
            
            return DatabaseContent(
                filename=filename,
                db_type='access',
                tables=tables,
                content="\n".join(content_parts),
                row_count=total_rows
            )

        except pyodbc.Error as e:
            return DatabaseContent(
                filename=filename,
                db_type='access',
                tables=[],
                content="",
                error=f"Access database error: {str(e)}"
            )

        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
    
    def _read_qvd(self, file_path: str, filename: str) -> DatabaseContent:
        """Read QlikView QVD file"""
        try:
            # Try multiple QVD libraries
            try:
                import qvd
                df = qvd.read(file_path)
            except ImportError:
                try:
                    from pyqvd import read_qvd
                    df = read_qvd(file_path)
                except ImportError:
                    return DatabaseContent(
                        filename=filename,
                        db_type='qvd',
                        tables=[],
                        content="",
                        error="No QVD library installed. Install with: pip install qvd or pip install pyqvd"
                    )
            
            # QVD files are single tables
            columns = list(df.columns)
            row_count = len(df)
            
            content_parts = [
                f"[QVD Data: {filename}]",
                f"Columns: {', '.join(columns)}",
                f"Rows: {row_count}",
                "",
                "Sample data:",
                " | ".join(columns)
            ]
            
            # Sample rows
            sample = df.head(self.max_rows)
            for _, row in sample.iterrows():
                content_parts.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
            
            # Column statistics
            content_parts.append("\nColumn Statistics:")
            for col in columns:
                try:
                    unique = df[col].nunique()
                    content_parts.append(f"  {col}: {unique} unique values")
                except:
                    pass
            
            return DatabaseContent(
                filename=filename,
                db_type='qvd',
                tables=[filename],
                content="\n".join(content_parts),
                row_count=row_count,
                metadata={'columns': columns}
            )
            
        except Exception as e:
            return DatabaseContent(
                filename=filename,
                db_type='qvd',
                tables=[],
                content="",
                error=f"Error reading QVD: {str(e)}"
            )
    
    def format_content(self, db_content: DatabaseContent) -> str:
        """Format database content for LLM context"""
        if db_content.error:
            return f"[Database Error - {db_content.filename}]: {db_content.error}"
        
        header = f"[Database: {db_content.filename}] ({db_content.db_type.upper()})"
        header += f" - {len(db_content.tables)} tables, {db_content.row_count} total rows"
        
        return f"{header}\n{db_content.content}"
