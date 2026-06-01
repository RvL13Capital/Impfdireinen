"""Entry point for Streamlit Community Cloud.

On https://share.streamlit.io, set the app's "Main file path" to
``streamlit_app.py`` (this file). Locally you can also run it directly::

    streamlit run streamlit_app.py

It simply launches the Phase-5 dashboard defined in ``vpts.dashboard.app``.
"""
from __future__ import annotations

from vpts.dashboard.app import main

main()
