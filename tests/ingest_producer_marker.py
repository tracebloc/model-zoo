"""The ``ingest_producer`` marker: deselected unless a ``-m`` expression names it.

A test carrying this marker compares the vendored category enum against a
checkout of the producer. Only one CI job has that checkout, so everywhere else
the test is DESELECTED -- reported in the summary as deselected, never collected
and skipped, and never run to fail on a missing input it was never meant to have.

Where it IS selected (``pytest -m ingest_producer``) it runs, and without a
producer checkout it FAILS: that job exists to look, and "could not look" there
is a red, not a pass.

Kept in its own file so ``tests/test_zoo_category_contract.py`` can drive these
exact hooks in a throwaway session instead of restating them. ``conftest.py``
re-exports them.
"""

MARKER = "ingest_producer"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{MARKER}: compares a vendored contract against a checkout of the producer; "
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
