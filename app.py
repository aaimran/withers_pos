#!/usr/bin/env python3
"""Dash app to browse WaveQLab3D station time series."""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, NamedTuple

try:
    from typing import TypedDict
except ImportError:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        TypedDict = dict

import plotly.graph_objects as go
from plotly.colors import qualitative as plotly_qual
from dash import ALL, Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# --- 1. SETUP & HELPER CLASSES ---

class TimeSeries(TypedDict):
    t: List[float]
    vx: List[float]
    vy: List[float]
    vz: List[float]

# Try to import ctx
try:
    from dash import ctx
except ImportError:
    ctx = None

# Regex patterns
FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<q>[^_]+)_(?P<r>[^_]+)_(?P<s>[^_]+)_(?P<block>block[^.]+)\.dat$",
    flags=re.IGNORECASE,
)
STATION_NAME_RE = re.compile(
    r"^(?P<dataset>.+?)_station_(?P<station>[A-Za-z0-9]+)\.dat$",
    flags=re.IGNORECASE,
)
NUM_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
XYZ_RE = re.compile(
    rf"^(?P<dataset>.+?)_(?P<x>{NUM_RE})_(?P<y>{NUM_RE})_(?P<z>{NUM_RE})\.dat$",
    flags=re.IGNORECASE,
)

class DatasetInfo(NamedTuple):
    path: Path
    dataset: str
    station: str

VARIANT_ELASTIC = "elastic"
VARIANT_ANELASTIC = "anelastic"

PML_TOKEN_RE = re.compile(r"^pml-(?P<mode>[^_]+)$", flags=re.IGNORECASE)
RES_TOKEN_RE = re.compile(r"^res-(?P<value>[^_]+)$", flags=re.IGNORECASE)
TEST_TOKEN_RE = re.compile(r"(?:^|[_-])test[-_]?(?P<id>\d+[a-z]?)", flags=re.IGNORECASE)
CG_VALUE_RE = re.compile(r"(?:^|[_-])cg-(?P<val>\d+)(?:[_-]|$)", flags=re.IGNORECASE)
ANELASTIC_GAMMA_RE = re.compile(r"anelastic[-_]gamma-(?P<val>[+-]?(?:\d+(?:\.\d+)?|\.\d+))", flags=re.IGNORECASE)
ELASTIC_RE = re.compile(r"(?:^|[_-])elastic(?:[_-]|$)", flags=re.IGNORECASE)
STENCIL_SET = {"traditional", "upwind", "upwind-drp"}

# --- 2. HELPER FUNCTIONS ---

def dataset_base_and_variant(dataset_name: str) -> Tuple[str, Optional[str]]:
    name = dataset_name
    if name.endswith('.dat'):
        name = name[:-4]

    # If filename ends with three numeric components (x_y_z) treat those as station coords
    parts = name.split('_')
    if len(parts) > 3 and all(re.match(r'^-?\d+(\.\d+)?$', p) for p in parts[-3:]):
        name = '_'.join(parts[:-3])

    # Try to pull out explicit variant tokens (anelastic/elastic) so callers get a clean base
    # Prefer the more specific anelastic-gamma token, falling back to plain 'elastic'
    m = ANELASTIC_GAMMA_RE.search(name)
    if m:
        # remove the matched token from the base name but keep separators
        base = ANELASTIC_GAMMA_RE.sub('_', name)
        base = re.sub(r'_+', '_', base).strip('_')
        return base, m.group('val')
    if ELASTIC_RE.search(name):
        base = ELASTIC_RE.sub('_', name)
        base = re.sub(r'_+', '_', base).strip('_')
        return base, 'elastic'

    return name, None


def parse_stencil_order_pml_ver(base: str) -> Optional[Tuple[str, str, str, str, str]]:
    parts = [p for p in base.split('_') if p]
    if len(parts) < 2:
        return None

    stencil = ""
    order = ""
    idx = -1
    for i, p in enumerate(parts[:-1]):
        if p.lower() in STENCIL_SET:
            stencil = p
            order = parts[i + 1]
            idx = i + 2
            break
    if not stencil or not order or idx < 0:
        return None

    res_value = ""
    pml_mode: Optional[str] = None
    rest: List[str] = []

    for p in parts[idx:]:
        if pml_mode is None:
            mres = RES_TOKEN_RE.match(p)
            if mres and not res_value:
                res_value = mres.group('value')
                continue
            m = PML_TOKEN_RE.match(p)
            if m:
                pml_mode = m.group('mode').lower()
                continue
        else:
            rest.append(p)

    if not pml_mode:
        return None

    if not rest:
        ver = ""
    elif rest[0].lower() == 'b' and len(rest) >= 2:
        ver = '_'.join(rest[1:])
    else:
        ver = '_'.join(rest)

    return stencil, order, res_value, pml_mode, ver

def parse_test_id(dataset: str) -> str:
    m = TEST_TOKEN_RE.search(dataset)
    return m.group('id') if m else ""

def parse_cg_value(dataset: str) -> str:
    m = CG_VALUE_RE.search(dataset)
    return m.group('val') if m else ""

def parse_variant_gamma(dataset: str) -> Tuple[Optional[str], Optional[str]]:
    m = ANELASTIC_GAMMA_RE.search(dataset)
    if m:
        return VARIANT_ANELASTIC, m.group('val')
    if ELASTIC_RE.search(dataset):
        return VARIANT_ELASTIC, "0.0"
    return None, None

def pml_label(pml_mode: str) -> str:
    mode = (pml_mode or "").lower()
    if mode == "off":
        return "0.0"
    if mode == "on":
        return "3.0"
    if mode == "60":
        return "6.0"
    # Try to extract leading numeric value (strip units like 'km' or 'm')
    m = re.match(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", mode)
    if m:
        return m.group(1)
    return pml_mode

def iter_dataset_files(data_dir: Path) -> List[Path]:
    patterns = ["*.dat"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(Path(p).resolve() for p in glob.glob(str(data_dir / pat)))
    files = [p for p in files if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files

def parse_dataset_info(path: Path) -> Optional[DatasetInfo]:
    if path.suffix.lower() != ".dat":
        return None
    stem = path.stem
    parts = stem.split('_')
    if len(parts) >= 3 and all(re.match(r'^-?\d+(\.\d+)?$', p) for p in parts[-3:]):
        station = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
        dataset = '_'.join(parts[:-3])
        return DatasetInfo(path=path, dataset=dataset, station=station)

    m2 = STATION_NAME_RE.match(path.name)
    if m2:
        dataset = m2.group("dataset")
        station = f"station_{m2.group('station')}"
        return DatasetInfo(path=path, dataset=dataset, station=station)

    m3 = XYZ_RE.match(path.name)
    if m3:
        dataset = m3.group("dataset")
        x, y, z = m3.group("x"), m3.group("y"), m3.group("z")
        station = f"{x}_{y}_{z}"
        return DatasetInfo(path=path, dataset=dataset, station=station)

    if len(parts) >= 5:
        dataset = "_".join(parts[:-4])
        q, r, s, block = parts[-4], parts[-3], parts[-2], parts[-1]
        if dataset and block.lower().startswith("block"):
            station = f"{q}_{r}_{s}_{block}"
            return DatasetInfo(path=path, dataset=dataset, station=station)
    return None

def load_timeseries(path: Path) -> TimeSeries:
    t, vx, vy, vz = [], [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 4: continue
            try:
                tt, vxx, vyy, vzz = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError: continue
            t.append(tt); vx.append(vxx); vy.append(vyy); vz.append(vzz)
    return {"t": t, "vx": vx, "vy": vy, "vz": vz}

def make_figure(station: str, selected: List[DatasetInfo], plot: str, properties: Dict[str, Dict[str, float]], height: int = 400) -> go.Figure:
    plot_meta = {
        "vx": (f"particle velocity in the x-direction at station {station}", "vx"),
        "vy": (f"particle velocity in the y-direction at station {station}", "vy"),
        "vz": (f"particle velocity in the z-direction at station {station}", "vz"),
    }
    if plot not in plot_meta:
        fig = go.Figure()
        fig.add_annotation(text="Invalid plot", showarrow=False)
        return fig

    title, y_col = plot_meta[plot]
    fig = go.Figure()
    datasets_in_order = sorted({i.dataset for i in selected})
    palette = list(plotly_qual.Plotly) or ["#1f77b4"]
    dataset_to_color = {ds: palette[idx % len(palette)] for idx, ds in enumerate(datasets_in_order)}

    for info in selected:
        df = load_timeseries(info.path)
        label = info.dataset
        color = dataset_to_color.get(label)
        props = properties.get(label, {})
        width = props.get('width', 2)
        opacity = props.get('opacity', 1)
        fig.add_trace(go.Scatter(x=df["t"], y=df[y_col], mode="lines", name=label,
                                 line=dict(color=color, width=width), opacity=opacity, showlegend=True))

    fig.update_layout(title=title, height=height, margin=dict(l=40, r=20, t=60, b=40),
                      xaxis_title="time (s)", yaxis_title="particle velocity (m/s)")
    return fig

def build_index(data_dir: Path) -> Tuple[List[DatasetInfo], List[str]]:
    infos: List[DatasetInfo] = []
    for path in iter_dataset_files(data_dir):
        info = parse_dataset_info(path)
        if info is not None:
            infos.append(info)
    stations = sorted({i.station for i in infos})
    return infos, stations

def group_by_station(infos: Iterable[DatasetInfo]) -> Dict[str, List[DatasetInfo]]:
    out: Dict[str, List[DatasetInfo]] = {}
    for i in infos:
        out.setdefault(i.station, []).append(i)
    for st in out:
        out[st].sort(key=lambda x: x.dataset)
    return out

# --- 3. GLOBAL INITIALIZATION (Executed on Import) ---

# Argument Parsing (handled safely for Gunicorn)
# We use argparse primarily to find the Data Dir, but fallback to environment/CWD
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None, help="Directory containing station .dat files")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8050)
parser.add_argument("--debug", action="store_true")

# In Gunicorn/Production, sys.argv might not be what we expect, so we catch errors
# or rely on defaults.
try:
    args, unknown = parser.parse_known_args()
except SystemExit:
    # If argparse fails (e.g. during build), just use defaults
    args = argparse.Namespace(data_dir=None, host="0.0.0.0", port=8050, debug=False)

# Locate Data Directory
script_dir = Path(__file__).resolve().parent
candidates = [
    Path.cwd() / "data",
    Path.cwd() / "waveqlab3d/simulation/plots",
    script_dir / "waveqlab3d/simulation/plots",
    script_dir / "data",
    script_dir,
]
# If --data-dir was passed, prioritize it
if args.data_dir:
    candidates.insert(0, Path(args.data_dir).expanduser().resolve())

data_dir = next((p.resolve() for p in candidates if p.exists()), candidates[-1].resolve())

# Initialize Data Index
all_infos, stations = build_index(data_dir)
by_station = group_by_station(all_infos)

# Initialize App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# CRITICAL: Expose server for Gunicorn
server = app.server

# Setup Defaults for Layout
initial_station = stations[0] if stations else ""
initial_infos = by_station.get(initial_station, [])
initial_selected_infos = [initial_infos[0]] if initial_infos else []
initial_plots = ["vx", "vy", "vz"]
initial_figs = [make_figure(initial_station, initial_selected_infos, p, {}, 400) for p in initial_plots]

# --- 4. LAYOUT DEFINITION ---

app.layout = dbc.Container(
    [
        dbc.Badge(f"Data dir: {data_dir}", color="secondary", className="mb-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Station viewer: Withers"),
                        html.Label("Station"),
                        dcc.Dropdown(
                            id="station-dropdown",
                            options=[{"label": s, "value": s} for s in stations],
                            value=initial_station,
                            multi=False, clearable=False, placeholder="Choose station",
                        ),
                        html.Hr(),
                        html.Label("Datasets"),
                        html.Button("Clear Dataset Selections", id="clear-dataset-button", style={"marginLeft": "10px"}),
                        dcc.Store(id="dataset-path-map", data={}),
                        dcc.Store(id="dataset-base-order", data=[]),
                        dcc.Store(id="dataset-selection-store", data={}),
                        html.Div(id="dataset-table-container"),
                        html.Hr(),
                        html.Div([
                            html.Label("Select timeseries plot(s):", style={"marginRight": "10px"}),
                            dcc.Checklist(
                                id="plot-checklist",
                                options=[{"label": "vx", "value": "vx"}, {"label": "vy", "value": "vy"}, {"label": "vz", "value": "vz"}],
                                value=initial_plots,
                                labelStyle={"display": "inline", "marginRight": "15px"}, style={"display": "inline"},
                            ),
                        ], style={"display": "flex", "alignItems": "center"}),
                        html.Hr(),
                        html.Label("Adjust line properties:"),
                        dcc.Store(id="line-properties-store", data={}),
                        html.Div(id="line-controls-container"),
                        html.Hr(),
                        html.Label("Adjust plot height:"),
                        dcc.Slider(id="plot-height-slider", min=200, max=800, value=400, step=50, marks={200:'200', 400:'400', 600:'600', 800:'800'}),
                    ],
                    style={"flex": "0 0 20%", "maxWidth": "20%"},
                ),
                dbc.Col(
                    [
                        html.Div(id="plot-container", children=[dcc.Graph(figure=fig) for fig in initial_figs])
                    ],
                    style={"flex": "0 0 80%", "maxWidth": "80%", "overflowY": "auto", "maxHeight": "95vh"},
                ),
            ],
            align="start",
        ),
    ],
    fluid=True,
    className="p-3",
)

# --- 5. CALLBACKS ---

@app.callback(
    [Output("dataset-table-container", "children"),
     Output("dataset-path-map", "data"),
     Output("dataset-base-order", "data")],
    [Input("station-dropdown", "value"),
     Input("dataset-selection-store", "data")]
)
def update_dataset_table(selected_station: str, selection_store: Dict):
    infos = by_station.get(selected_station or "", [])
    stencil_to_variants: Dict[str, Dict[Tuple[str, str, str, str, str], Dict[str, str]]] = {}
    base_to_cg: Dict[str, str] = {}
    base_to_gamma: Dict[str, str] = {}
    
    for info in infos:
        base, _ = dataset_base_and_variant(info.dataset)
        test_id = parse_test_id(info.dataset)
        variant, gamma = parse_variant_gamma(info.dataset)
        if variant not in (VARIANT_ELASTIC, VARIANT_ANELASTIC):
            continue
        parsed = parse_stencil_order_pml_ver(base)
        if parsed is None:
            continue
        stencil, order, res, pml_mode, ver = parsed
        key = (test_id, order, res, pml_mode, ver)
        if stencil not in stencil_to_variants: stencil_to_variants[stencil] = {}
        if key not in stencil_to_variants[stencil]: stencil_to_variants[stencil][key] = {}
        stencil_to_variants[stencil][key].setdefault(variant, str(info.path))
        base_key = f"test{test_id}_{stencil}_{order}" if test_id else f"{stencil}_{order}"
        if res:
            base_key += f"_res-{res}"
        base_key += f"_pml-{pml_mode}"
        if ver:
            base_key += f"_{ver}"
        if base_key not in base_to_cg:
            base_to_cg[base_key] = parse_cg_value(info.dataset)
        if variant == VARIANT_ANELASTIC and gamma and base_key not in base_to_gamma:
            base_to_gamma[base_key] = gamma

    sorted_stencils = sorted(stencil_to_variants.keys(), key=lambda s: ['traditional', 'upwind', 'upwind-drp'].index(s) if s in ['traditional', 'upwind', 'upwind-drp'] else 999)
    grouped = {}
    base_order = []

    def sort_key(k: Tuple[str, str, str, str, str]) -> Tuple[int, str, int, float, int, str]:
        test_s, order_s, res_s, pml_s, ver_s = k
        mtest = re.match(r"^(\d+)([a-z]?)$", test_s or "")
        if mtest:
            test_i = int(mtest.group(1))
            test_suffix = mtest.group(2)
        else:
            test_i = 10**9
            test_suffix = ""
        try:
            order_i = int(order_s)
        except ValueError:
            order_i = 10**9
        try:
            res_f = float(res_s) if res_s else float('inf')
        except ValueError:
            res_f = float('inf')
        pml_rank = 0 if pml_s.lower() == 'off' else 1
        return (test_i, test_suffix, order_i, res_f, pml_rank, ver_s)
    
    for stencil in sorted_stencils:
        for key in sorted(stencil_to_variants[stencil].keys(), key=sort_key):
            test_id, order, res, pml_mode, ver = key
            base = f"test{test_id}_{stencil}_{order}" if test_id else f"{stencil}_{order}"
            if res:
                base += f"_res-{res}"
            base += f"_pml-{pml_mode}"
            if ver:
                base += f"_{ver}"
            grouped[base] = stencil_to_variants[stencil][key]
            base_order.append(base)

    has_any_selection = False
    for base in base_order:
        saved = selection_store.get(base, {}) if isinstance(selection_store, dict) else {}
        if bool(saved.get(VARIANT_ELASTIC)) or bool(saved.get(VARIANT_ANELASTIC)):
            has_any_selection = True; break

    if not has_any_selection and base_order:
        first_base = base_order[0]
        selection_store = dict(selection_store or {})
        selection_store.setdefault(first_base, {})
        if grouped[first_base].get(VARIANT_ELASTIC):
            selection_store[first_base][VARIANT_ELASTIC] = True
        elif grouped[first_base].get(VARIANT_ANELASTIC):
            selection_store[first_base][VARIANT_ANELASTIC] = True

    header = html.Thead(html.Tr([html.Th("Test"), html.Th("CG"), html.Th("Stencil"), html.Th("Order"), html.Th("Res"), html.Th("PML"), html.Th("Ver."), html.Th("Elastic"), html.Th("Anelastic")]))

    # Build table rows directly from base_order. This also acts as a fallback when
    # stencil-based grouping yields no results (so datasets still show up).
    rows = []
    for base in base_order:
        variants = grouped.get(base, {})
        elastic_path = variants.get(VARIANT_ELASTIC)
        anelastic_path = variants.get(VARIANT_ANELASTIC)
        saved = selection_store.get(base, {}) if isinstance(selection_store, dict) else {}

        elastic_cell = html.Td(
            dcc.Checklist(
                id={"type": "dataset-elastic", "base": base},
                options=[{"label": "0.0", "value": "on", "disabled": elastic_path is None}],
                value=["on"] if bool(saved.get(VARIANT_ELASTIC)) and elastic_path else [],
            ),
            style={"textAlign": "right"},
        )
        gamma_label = base_to_gamma.get(base, "")
        anelastic_cell = html.Td(
            dcc.Checklist(
                id={"type": "dataset-anelastic", "base": base},
                options=[{"label": gamma_label or "anelastic", "value": "on", "disabled": anelastic_path is None}],
                value=["on"] if bool(saved.get(VARIANT_ANELASTIC)) and anelastic_path else [],
            ),
            style={"textAlign": "left"},
        )

        # Try to extract display tokens from base string for nicer columns
        test_val = parse_test_id(base)
        cg_val = base_to_cg.get(base, "")

        m_pml = re.search(r"pml-(?P<mode>[^_]+)", base, flags=re.IGNORECASE)
        pml_val = pml_label(m_pml.group('mode')) if m_pml else "-"
        m_res = re.search(r"res-(?P<value>[^_]+)", base, flags=re.IGNORECASE)
        res_display = m_res.group('value') if m_res else "-"
        m_stencil = re.search(r"(traditional|upwind|upwind-drp)_([^_]+)", base)
        if m_stencil:
            stencil_val = m_stencil.group(1)
            order_val = m_stencil.group(2)
        else:
            stencil_val = "-"
            order_val = "-"

        ver_val = ""

        row_children = [html.Td(test_val, style={"textAlign": "center"}), html.Td(cg_val, style={"textAlign": "center"}), html.Td(stencil_val, style={"textAlign": "center"}), html.Td(order_val, style={"textAlign": "center"}), html.Td(res_display, style={"textAlign": "center"}), html.Td(pml_val, style={"textAlign": "center"}), html.Td(ver_val, style={"textAlign": "center"}), elastic_cell, anelastic_cell]
        rows.append(html.Tr(row_children))

    return dbc.Table([header, html.Tbody(rows)], bordered=True, hover=True, size="sm", responsive=True), grouped, base_order

@app.callback(Output("dataset-selection-store", "data"), Input("clear-dataset-button", "n_clicks"), prevent_initial_call=True)
def clear_dataset_selections(n_clicks): return {}

@app.callback(Output("line-properties-store", "data"), [Input({"type": "width", "dataset": ALL}, "value"), Input({"type": "opacity", "dataset": ALL}, "value")], State("line-properties-store", "data"), prevent_initial_call=True)
def update_line_properties(width_values, opacity_values, current_props):
    if not current_props: current_props = {}
    triggered = ctx.triggered if ctx else []
    if triggered:
        prop_id = triggered[0]['prop_id']
        import json
        id_dict = json.loads(prop_id.split('.')[0])
        ds, typ, val = id_dict['dataset'], id_dict['type'], triggered[0]['value']
        if ds not in current_props: current_props[ds] = {}
        current_props[ds][typ] = val
    return current_props

@app.callback(
    [Output("plot-container", "children"), Output("line-controls-container", "children")],
    [Input("station-dropdown", "value"), Input("plot-checklist", "value"), Input({"type": "dataset-elastic", "base": ALL}, "value"), Input({"type": "dataset-anelastic", "base": ALL}, "value"), Input("plot-height-slider", "value")],
    [State("dataset-path-map", "data"), State("dataset-base-order", "data"), State("line-properties-store", "data")]
)
def update_plot(station, plots_selected, elastic_values, anelastic_values, plot_height, path_map, base_order, properties):
    selected_infos = []
    if base_order and isinstance(path_map, dict):
        for idx, base in enumerate(base_order):
            variants = path_map.get(base, {})
            if idx < len(elastic_values) and elastic_values[idx]:
                p = variants.get(VARIANT_ELASTIC)
                if p:
                    info = parse_dataset_info(Path(p))
                    if info:
                        selected_infos.append(info)
            if idx < len(anelastic_values) and anelastic_values[idx]:
                p = variants.get(VARIANT_ANELASTIC)
                if p:
                    info = parse_dataset_info(Path(p))
                    if info:
                        selected_infos.append(info)

    figs = [make_figure(station or "", selected_infos, p, properties, plot_height) for p in plots_selected or []]
    
    selected_datasets = sorted({info.dataset for info in selected_infos})
    if selected_datasets:
        rows = []
        for ds in selected_datasets:
            props = properties.get(ds, {})
            rows.append(html.Tr([
                html.Td(ds),
                html.Td(dcc.Slider(id={"type": "width", "dataset": ds}, min=1, max=5, value=props.get('width', 2), step=1, marks={1:'1',3:'3',5:'5'})),
                html.Td(dcc.Slider(id={"type": "opacity", "dataset": ds}, min=0, max=1, value=props.get('opacity', 1), step=0.1, marks={0:'0',0.5:'0.5',1:'1'}))
            ]))
        controls = dbc.Table([html.Thead(html.Tr([html.Th("Dataset"), html.Th("Line Width"), html.Th("Line Opacity")])), html.Tbody(rows)], bordered=True, size="sm")
    else:
        controls = html.Div()
        
    return [dcc.Graph(figure=fig) for fig in figs], controls

# --- 6. EXECUTION ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)