from pathlib import Path
import pytest


@pytest.mark.integration
def test_dashboard_demo_and_results():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception
    app.slider[1].set_value(1)
    app.button[0].click().run(timeout=120)
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4
    assert "experiment" in app.session_state
    assert len(app.tabs) == 4
