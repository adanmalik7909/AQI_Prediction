"""
tests/test_app_smoke.py
-------------------------
Runs the whole Streamlit dashboard headlessly and asserts it renders without
an uncaught exception.

This is the only test that exercises the app top to bottom - layout, charts,
SHAP tabs and all. It runs against the LOCAL model bundles (AQI_LOCAL_MODELS=1)
so it needs no Hopsworks credentials.

Run:  python -m pytest tests/test_app_smoke.py -v
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "webapp", "app.py")
BUNDLE = os.path.join(ROOT, "trained_models", "target_24h",
                      "registry_bundle", "metadata.json")


@pytest.fixture(scope="module")
def app_run():
    if not os.path.exists(BUNDLE):
        pytest.skip("no local model bundle - run train_models.py first")

    pytest.importorskip("streamlit.testing.v1",
                        reason="streamlit too old for AppTest")
    from streamlit.testing.v1 import AppTest

    # Force the local-bundle path so the test never depends on the network.
    os.environ["AQI_LOCAL_MODELS"] = "1"

    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    return at


def page_text(app_run):
    """All markdown on the page EXCEPT the CSS block.

    The stylesheet defines the same class names the assertions look for
    (.prov-badge, .fc-value), so counting them across raw markdown would match
    the definitions as well as the rendered elements.
    """
    return " ".join(str(m.value) for m in app_run.markdown
                    if "<style>" not in str(m.value))


def test_app_renders_without_exception(app_run):
    problems = [str(e.value) for e in app_run.exception]
    assert not problems, "dashboard raised:\n" + "\n".join(problems)


def test_app_renders_its_main_sections(app_run):
    """Guards against a silent half-render: a broken chart or missing feature
    can leave the page technically exception-free but largely empty."""
    assert len(app_run.tabs) == 3, "expected one SHAP tab per horizon"
    assert len(app_run.dataframe) >= 1, "accuracy disclosure table missing"

    text = page_text(app_run)
    for expected in ["3-Day AQI Forecast", "Current Conditions",
                     "Why These Predictions?", "System & Pipeline"]:
        assert expected in text, f"section missing from page: {expected}"


    # The forecast cards are custom HTML (fc-value) rather than st.metric, so
    # counting metrics would silently pass on an empty page.
    assert text.count("fc-value") == 3, "expected three forecast cards"


def test_app_declares_its_model_and_feature_source(app_run):
    """The Hopsworks-first requirement is only demonstrable if the page states
    which source actually served the run. Assert the badge is present and that
    it is one of the two honest verdicts, not a hardcoded claim."""
    text = page_text(app_run)

    assert "Model source:" in text, "model provenance badge missing"
    assert "Feature source:" in text, "feature provenance badge missing"

    # prov-ok = Hopsworks, prov-warn = fallback. Exactly two badges render, and
    # each must carry one of the two states.
    assert text.count("prov-badge") == 2, "expected exactly two source badges"
    assert "prov-ok" in text or "prov-warn" in text, \
        "badge rendered without a source verdict"




def test_app_reports_accuracy_honestly(app_run):
    """The dashboard must not present a bare R2. The naive-baseline comparison
    and the backtest score are what make the number interpretable."""
    text = page_text(app_run).lower()
    assert "backtest" in text, "backtest score not disclosed"
    assert "naive" in text, "naive baseline not disclosed"



if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
