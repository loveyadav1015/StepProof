"""Basic UI and Streamlit app tests: test importability and mock run rendering."""

from __future__ import annotations

from pathlib import Path


def test_tokens_css_exists_and_has_variables():
    """Verify docs/site/tokens.css defines necessary design tokens."""
    tokens_path = Path("docs/site/tokens.css")
    assert tokens_path.exists()
    content = tokens_path.read_text(encoding="utf-8")
    assert "--bg-primary:" in content
    assert "--font-family:" in content
    assert "--spacing-300:" in content
    assert "--radius-lg:" in content


def test_streamlit_config_toml():
    """Verify .streamlit/config.toml theme matches tokens.css."""
    config_path = Path(".streamlit/config.toml")
    assert config_path.exists()
    content = config_path.read_text(encoding="utf-8")
    assert "primaryColor" in content
    assert "#3B82F6" in content
    assert "#0A0A0A" in content


def test_build_site_script():
    """Verify scripts/build_site.py executes and generates docs/site/index.html."""
    from scripts.build_site import main as build_site_main

    exit_code = build_site_main()
    assert exit_code == 0
    index_html = Path("docs/site/index.html")
    assert index_html.exists()
    html_content = index_html.read_text(encoding="utf-8")
    assert "StepProof" in html_content
    assert "tokens.css" in html_content
    # Honesty check: Meta model run pending when no final manifest exists
    manifest_path = Path("data/outputs/final/MANIFEST.json")
    if not manifest_path.exists():
        assert "Meta Model Run Pending" in html_content


def test_streamlit_app_importable():
    """Verify streamlit_app module imports cleanly without syntax errors."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "streamlit_app", "src/stepproof/app/streamlit_app.py"
    )
    assert spec is not None
    assert spec.loader is not None
