from pathlib import Path
import re

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent

PATTERN = re.compile(r"^(qx|px|qy|py)(\d+)\.html$")

ORDER = ("qx", "px", "qy", "py")

LABELS = {
    1: "(nux, nuy) = (0.280, 0.310)",
    2: "(nux, nuy) = (0.168, 0.201)"
}

PLOT_LABELS = {
    "qx": "qx=0",
    "px": "px=0",
    "qy": "qy=0",
    "py": "py=0",
}

GRAPH_DIV_PATTERN = re.compile(r'(<div id="[^"]+" class="plotly-graph-div" style=")height:100%; width:100%;("></div>)')


def locate() -> dict[int, dict[str, Path]]:
    plots: dict[int, dict[str, Path]] = {}
    for path in ROOT.glob("*.html"):
        match = PATTERN.match(path.name)
        if not match:
            continue
        plot_name, case_text = match.groups()
        case = int(case_text)
        plots.setdefault(case, {})[plot_name] = path
    return dict(sorted(plots.items()))


def select(case: int) -> None:
    st.session_state.case = case


def set_plot(plot_name: str) -> None:
    st.session_state.plot_name = plot_name


def render(path: Path, plot_height: int) -> str:
    html = path.read_text(encoding="utf-8")
    html = html.replace(
        "<head><meta charset=\"utf-8\" /></head>",
        (
            "<head><meta charset=\"utf-8\" />"
            "<style>html, body { margin: 0; height: 100%; } "
            ".plotly-graph-div { min-height: 100%; }</style></head>"
        ),
        1,
    )
    html, count = GRAPH_DIV_PATTERN.subn(rf"\1height:{plot_height}px; width:100%;\2", html, count=1)
    if count == 0:
        st.warning(f"Could not adjust embedded height for {path.name}.")
    return html


table = locate()

st.set_page_config(page_title="4D Henon mapping bounding set 3D projections", layout="wide")
st.title("4D Henon mapping bounding set 3D projections")

if not table:
    st.error("No files matching qxN.html, pxN.html, qyN.html, or pyN.html were found.")
    st.stop()

cases = list(table)
names = {name for case in table.values() for name in case}

if "case" not in st.session_state or st.session_state.case not in table:
    st.session_state.case = cases[0]

if "plot_name" not in st.session_state or st.session_state.plot_name not in names:
    st.session_state.plot_name = next(
        name for name in ORDER if name in table[st.session_state.case]
    )

with st.sidebar:
    st.header("Controls")
    st.subheader("Case")
    for case in cases:
        st.button(
            LABELS.get(case, f"Case {case}"),
            key=f"case_{case}",
            type="primary" if case == st.session_state.case else "secondary",
            use_container_width=True,
            on_click=select,
            args=(case,),
        )

current = table[st.session_state.case]

if st.session_state.plot_name not in current:
    st.session_state.plot_name = next(name for name in ORDER if name in current)

with st.sidebar:
    st.subheader("Plot")
    names = [name for name in ORDER if name in current]
    for name in names:
        st.button(
            PLOT_LABELS[name],
            key=f"plot_{name}",
            type="primary" if name == st.session_state.plot_name else "secondary",
            use_container_width=True,
            on_click=set_plot,
            args=(name,),
        )

path = current[st.session_state.plot_name]
st.caption(f"Showing `{path.name}`")

plot_height = 800
html = render(path, plot_height)
components.html(html, height=plot_height, scrolling=True)
