import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

from engine.support_matrix import (  # noqa: E402
    README_TABLE_END,
    README_TABLE_START,
    REFUSALS,
    TASK_ENGINES,
    markdown_table,
)


def test_readme_carries_the_generated_table():
    readme = (BASE_DIR / "README.md").read_text(encoding="utf-8")
    start = readme.index(README_TABLE_START) + len(README_TABLE_START)
    end = readme.index(README_TABLE_END)
    assert readme[start:end].strip() == markdown_table()


def test_every_refusal_names_an_engine_the_task_runs_on():
    for task, engine, *_ in REFUSALS:
        assert engine in TASK_ENGINES[task], (task, engine)
