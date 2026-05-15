"""
app.py — Streamlit Cloud entry point
Set Main file path to: app.py
"""
import runpy, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "streamlit_app.py"),
    run_name="__main__"
)
