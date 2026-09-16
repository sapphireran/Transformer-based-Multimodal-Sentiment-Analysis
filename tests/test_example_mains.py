"""Smoke-run the example entry points that do not train."""

from examples import (
    inspect_pickle_schema,
    metrics_walkthrough,
    results_tables,
    synthetic_data,
)


def test_synthetic_main():
    assert synthetic_data.main() == 0


def test_schema_main_without_file():
    assert inspect_pickle_schema.main([]) == 0


def test_metrics_main():
    assert metrics_walkthrough.main() == 0


def test_results_tables_main():
    assert results_tables.main() == 0
