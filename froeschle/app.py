from pathlib import Path
import re

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent

PATTERN = re.compile(r"^(fqx|fpx|fqy|fpy)\.html$")

ORDER = ("fqx", "fpx", "fqy", "fpy")

PLOT_LABELS = {
    "fqx": "qx=0",
    "fpx": "px=0",
    "fqy": "qy=0",
    "fpy": "py=0",
}

GRAPH_DIV_PATTERN = re.compile(r'(<div id="[^"]+" class="plotly-graph-div" style=")(.*?)("></div>)')
LAYOUT_SIZE_PATTERN = re.compile(r'"width":\s*\d+,\s*"height":\s*\d+')


def locate() -> dict[str, Path]:
    plots: dict[str, Path] = {}
    for path in ROOT.glob("*.html"):
        match = PATTERN.match(path.name)
        if not match:
            continue
        plots[match.group(1)] = path
    return {name: plots[name] for name in ORDER if name in plots}


def set_plot(plot_name: str) -> None:
    st.session_state.plot_name = plot_name


def render(path: Path, plot_height: int) -> str:
    html = path.read_text(encoding="utf-8")
    html = html.replace(
        "<head><meta charset=\"utf-8\" /></head>",
        (
            "<head><meta charset=\"utf-8\" />"
            "<style>html, body { margin: 0; height: 100%; } "
            "body > div { height: 100%; } "
            ".js-plotly-plot, .plot-container, .plotly, .plotly-graph-div "
            f"{{ width: 100% !important; height: {plot_height}px !important; }}</style></head>"
        ),
        1,
    )
    html, count = GRAPH_DIV_PATTERN.subn(rf'\1height:{plot_height}px; width:100%;\3', html, count=1)
    if count == 0:
        html = LAYOUT_SIZE_PATTERN.sub(f'"width": 1800, "height": {plot_height}', html, count=1)
    return html


table = locate()

st.set_page_config(page_title="Froeschle mapping bounding set 3D projections", layout="wide")
st.title("Froeschle mapping bounding set 3D projections")

if not table:
    st.error("No files matching fqx.html, fpx.html, fqy.html, or fpy.html were found.")
    st.stop()

if "plot_name" not in st.session_state or st.session_state.plot_name not in table:
    st.session_state.plot_name = next(name for name in ORDER if name in table)

with st.sidebar:
    st.header("Controls")
    st.subheader("Plot")
    for name in table:
        st.button(
            PLOT_LABELS[name],
            key=f"plot_{name}",
            type="primary" if name == st.session_state.plot_name else "secondary",
            use_container_width=True,
            on_click=set_plot,
            args=(name,),
        )

path = table[st.session_state.plot_name]
st.caption(f"Showing `{path.name}`")

plot_height = 1600
html = render(path, plot_height)
components.html(html, height=plot_height, scrolling=True)
