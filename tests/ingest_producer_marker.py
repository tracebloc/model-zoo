"""The ``ingest_producer`` marker: deselected unless a ``-m`` expression names it.

A test carrying this marker reads a checkout of the producer (the ingestor):
the stability-selection template is checked on its first-run example data. Only
a run that names the marker selects these tests, so everywhere else they are
DESELECTED -- reported in the summary as deselected, never collected and
skipped, and never run to fail on a missing input they were never meant to have.

Where they ARE selected (``pytest -m ingest_producer``) they run, and without a
producer checkout they FAIL: such a run exists to look, and "could not look"
there is a red, not a pass.

Kept in its own file so ``tests/test_zoo_category_contract.py`` can drive these
exact hooks in a throwaway session instead of restating them. ``conftest.py``
re-exports them.
"""

MARKER = "ingest_producer"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{MARKER}: reads a checkout of the producer (the ingestor); "
        f"deselected unless `-m` names it",
    )


def pytest_collection_modifyitems(config, items):
    if MARKER in (config.getoption("markexpr") or ""):
        return
    keep, drop = [], []
    for item in items:
        (drop if item.get_closest_marker(MARKER) else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
