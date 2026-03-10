from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from scipy.stats import gaussian_kde


DATA_PATH = Path(__file__).resolve().parent / "student_productivity_distraction_dataset_20000.csv"
BASE_COLUMNS = [
    "age",
    "gender",
    "study_hours_per_day",
    "sleep_hours",
    "phone_usage_hours",
    "productivity_score",
    "focus_score",
    "stress_level",
]
SCATTER_DENSITY_BINS_X = 70
SCATTER_DENSITY_BINS_Y = 70
LOW_DENSITY_QUANTILE = 0.05


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[BASE_COLUMNS].copy()
    df["main_activity_time"] = (
        df["study_hours_per_day"] + df["sleep_hours"] + df["phone_usage_hours"]
    )

    totals = df["main_activity_time"]
    df = df.loc[totals > 0].copy()
    totals = df["main_activity_time"]

    df["study_ratio"] = df["study_hours_per_day"] / totals
    df["sleep_ratio"] = df["sleep_hours"] / totals
    df["phone_ratio"] = df["phone_usage_hours"] / totals
    return df


def make_range(values: pd.Series, round_digits: int = 2) -> list[float]:
    return [round(float(values.min()), round_digits), round(float(values.max()), round_digits)]


def make_step_aligned_range(values: pd.Series, step: float, round_digits: int) -> list[float]:
    min_value = float(values.min())
    max_value = float(values.max())
    low = np.floor(min_value / step) * step
    high = np.ceil(max_value / step) * step
    return [round(low, round_digits), round(high, round_digits)]


def slider_marks(min_value: float, max_value: float, digits: int = 0) -> dict[float, str]:
    return {
        round(min_value, digits): f"{min_value:.{digits}f}",
        round(max_value, digits): f"{max_value:.{digits}f}",
    }


def slider_marks_with_selection(
    min_value: float, max_value: float, selected_range: list[float] | None, digits: int = 0
) -> dict[float, str]:
    marks = slider_marks(min_value, max_value, digits)
    if not selected_range:
        return marks

    low = round(float(selected_range[0]), digits)
    high = round(float(selected_range[1]), digits)
    marks[low] = f"{low:.{digits}f}"
    marks[high] = f"{high:.{digits}f}"
    return dict(sorted(marks.items()))


def apply_filters(
    df: pd.DataFrame,
    genders: list[str],
    age_range: list[float],
    main_activity_range: list[float],
    productivity_range: list[float],
) -> pd.DataFrame:
    if not genders:
        return df.iloc[0:0]

    filtered = df[df["gender"].isin(genders)]
    filtered = filtered[
        (filtered["age"] >= age_range[0])
        & (filtered["age"] <= age_range[1])
        & (filtered["main_activity_time"] >= main_activity_range[0])
        & (filtered["main_activity_time"] <= main_activity_range[1])
        & (filtered["productivity_score"] >= productivity_range[0])
        & (filtered["productivity_score"] <= productivity_range[1])
    ]
    return filtered


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=40, r=20, t=55, b=40),
        annotations=[
            dict(
                text="No data under current filters",
                showarrow=False,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                font=dict(size=14, color="#666666"),
            )
        ],
    )
    return fig


TERNARY_HEIGHT = float(np.sqrt(3) / 2)
TERNARY_GRID_BINS = 120


def ternary_to_cartesian(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Vertex mapping: Sleep(a)=top, Study(b)=left, Phone(c)=right.
    x = c + 0.5 * a
    y = a * TERNARY_HEIGHT
    return x, y


def build_ternary_binned_matrices(
    df: pd.DataFrame, bins: int = TERNARY_GRID_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = df["sleep_ratio"].to_numpy(dtype=float)
    b = df["study_ratio"].to_numpy(dtype=float)
    c = df["phone_ratio"].to_numpy(dtype=float)
    values = df["productivity_score"].to_numpy(dtype=float)

    x, y = ternary_to_cartesian(a, b, c)
    x_edges = np.linspace(0.0, 1.0, bins + 1)
    y_edges = np.linspace(0.0, TERNARY_HEIGHT, bins + 1)

    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    y_idx = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    valid = np.isfinite(values)
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    values = values[valid]

    flat_idx = y_idx * bins + x_idx
    counts = np.bincount(flat_idx, minlength=bins * bins).reshape(bins, bins)
    sums = np.bincount(flat_idx, weights=values, minlength=bins * bins).reshape(bins, bins)

    mean_productivity = np.full((bins, bins), np.nan, dtype=float)
    np.divide(sums, counts, out=mean_productivity, where=counts > 0)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)
    a_center = yy / TERNARY_HEIGHT
    c_center = xx - 0.5 * a_center
    b_center = 1.0 - a_center - c_center
    inside_triangle = (a_center >= -1e-9) & (b_center >= -1e-9) & (c_center >= -1e-9)
    occupied = counts > 0
    valid_cells = inside_triangle & occupied

    mean_productivity[~valid_cells] = np.nan
    density_counts = counts.astype(float)
    density_counts[~valid_cells] = np.nan

    return x_centers, y_centers, mean_productivity, density_counts


def format_ternary_heatmap_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0, 0.5, 0.0],
            y=[0.0, 0.0, TERNARY_HEIGHT, 0.0],
            mode="lines",
            line=dict(color="#4f4f4f", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=40, r=20, t=55, b=45),
        annotations=[
            dict(
                text="Sleep Ratio",
                x=0.5,
                y=TERNARY_HEIGHT + 0.06,
                xref="x",
                yref="y",
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                text="Study Ratio",
                x=-0.02,
                y=-0.04,
                xref="x",
                yref="y",
                xanchor="left",
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                text="Phone Ratio",
                x=1.02,
                y=-0.04,
                xref="x",
                yref="y",
                xanchor="right",
                showarrow=False,
                font=dict(size=12),
            ),
        ],
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-0.03, 1.03],
        fixedrange=True,
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-0.06, TERNARY_HEIGHT + 0.08],
        scaleanchor="x",
        scaleratio=1,
        fixedrange=True,
    )
    return fig


def log1p_for_heatmap(values: np.ndarray) -> np.ndarray:
    logged = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return logged

    logged[valid] = np.log1p(np.maximum(values[valid], 0.0))
    return logged


def build_ternary_productivity(df: pd.DataFrame) -> go.Figure:
    title = "Main Activity Time Allocation vs Productivity"
    if df.empty:
        return empty_figure(title)

    x_centers, y_centers, mean_productivity, density_counts = build_ternary_binned_matrices(df)
    productivity_hover = np.stack([mean_productivity, density_counts], axis=-1)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=mean_productivity,
            colorscale="Viridis_r",
            colorbar=dict(title="Productivity"),
            customdata=productivity_hover,
            hovertemplate=(
                "Mean Productivity: %{customdata[0]:.2f}<br>"
                "Samples in bin: %{customdata[1]:.0f}<extra></extra>"
            ),
            showscale=True,
            zsmooth=False,
            hoverongaps=False,
        )
    )
    return format_ternary_heatmap_layout(fig, title)


def build_ternary_density(df: pd.DataFrame) -> go.Figure:
    title = "Distribution of Main Activity Time Allocation"
    if df.empty:
        return empty_figure(title)

    x_centers, y_centers, _mean_productivity, density_counts = build_ternary_binned_matrices(df)
    density_counts_log = log1p_for_heatmap(density_counts)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=density_counts_log,
            colorscale="Blues",
            colorbar=dict(title="Density", showticklabels=False),
            customdata=density_counts,
            hovertemplate="Samples in bin: %{customdata:.0f}<extra></extra>",
            showscale=True,
            zsmooth=False,
            hoverongaps=False,
        )
    )
    return format_ternary_heatmap_layout(fig, title)


def build_density_panel(df: pd.DataFrame, x_col: str, x_label: str) -> go.Figure:
    title = f"{x_label} vs Productivity"
    if df.empty:
        return empty_figure(title)

    valid = df[[x_col, "productivity_score"]].dropna()
    if valid.empty:
        return empty_figure(title)

    x_values = valid[x_col].to_numpy(dtype=float)
    y_values = valid["productivity_score"].to_numpy(dtype=float)

    counts, x_edges, y_edges = np.histogram2d(
        x_values, y_values, bins=[SCATTER_DENSITY_BINS_X, SCATTER_DENSITY_BINS_Y]
    )
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    density_counts = counts.T.astype(float)
    density_counts[density_counts == 0] = np.nan

    non_zero_density = density_counts[np.isfinite(density_counts)]
    if non_zero_density.size > 0 and np.unique(non_zero_density).size > 1:
        low_density_cutoff = float(np.quantile(non_zero_density, LOW_DENSITY_QUANTILE))
        density_counts[density_counts <= low_density_cutoff] = np.nan

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=density_counts,
            customdata=counts.T,
            colorscale="Blues",
            colorbar=dict(title="Density", showticklabels=False),
            hovertemplate=(
                f"{x_label}: %{{x:.2f}}<br>"
                "Productivity: %{y:.2f}<br>"
                "Samples in bin: %{customdata:.0f}<extra></extra>"
            ),
            showscale=True,
            hoverongaps=False,
            zsmooth=False,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=x_values,
            y=y_values,
            mode="markers",
            marker=dict(size=3, color="rgba(20,20,20,0.20)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=45, r=20, t=55, b=45),
        xaxis_title=f"{x_label} (hours/day)",
        yaxis_title="Productivity Score",
    )
    return fig


def build_focus_distribution(df: pd.DataFrame) -> go.Figure:
    title = "Distribution of Focus Score"
    if df.empty:
        return empty_figure(title)

    values = df["focus_score"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            histnorm="probability density",
            xbins=dict(size=1),
            name="Histogram",
            opacity=0.65,
            marker=dict(color="#6aa57a"),
        )
    )

    if np.unique(values).size > 1:
        x_grid = np.linspace(values.min(), values.max(), 220)
        kde = gaussian_kde(values)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=kde(x_grid),
                mode="lines",
                line=dict(color="#4c78a8", width=2),
                name="KDE",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        barmode="overlay",
        margin=dict(l=45, r=20, t=55, b=45),
        xaxis_title="Focus Score",
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_stress_distribution(df: pd.DataFrame) -> go.Figure:
    title = "Distribution of Stress Level"
    if df.empty:
        return empty_figure(title)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["stress_level"],
            histnorm="probability density",
            xbins=dict(start=0.5, end=10.5, size=1),
            marker=dict(color="#c76b6b"),
            name="Density",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=45, r=20, t=55, b=45),
        xaxis_title="Stress Level",
        yaxis_title="Density",
        bargap=0.08,
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1, range=[0.5, 10.5])
    return fig


DATA_DF = load_data(DATA_PATH)
TOTAL_RECORDS = len(DATA_DF)

raw_genders = list(pd.unique(DATA_DF["gender"].dropna()))
preferred_gender_order = ["Male", "Female", "Other"]
GENDER_FILTER_OPTIONS = [g for g in preferred_gender_order if g in raw_genders] + [
    g for g in raw_genders if g not in preferred_gender_order
]
if not GENDER_FILTER_OPTIONS:
    GENDER_FILTER_OPTIONS = preferred_gender_order.copy()
DEFAULT_GENDERS = GENDER_FILTER_OPTIONS.copy()
DEFAULT_AGE_RANGE = [int(DATA_DF["age"].min()), int(DATA_DF["age"].max())]
DEFAULT_MAIN_ACTIVITY_RANGE = make_step_aligned_range(
    DATA_DF["main_activity_time"], step=0.1, round_digits=1
)
DEFAULT_PRODUCTIVITY_RANGE = make_range(DATA_DF["productivity_score"], round_digits=0)


app = Dash(__name__)
app.title = "Student Productivity Dashboard"
server = app.server

control_style = {"display": "flex", "flexDirection": "column", "gap": "6px", "minWidth": "250px", "flex": "1 1 250px"}
row_style = {"display": "flex", "flexWrap": "wrap", "gap": "14px", "marginBottom": "16px"}
top_card_style = {"flex": "1 1 420px", "minWidth": "340px", "border": "1px solid #e5e5e5", "borderRadius": "10px", "padding": "8px"}
middle_card_style = {"flex": "1 1 320px", "minWidth": "280px", "border": "1px solid #e5e5e5", "borderRadius": "10px", "padding": "8px"}
bottom_card_style = {"flex": "1 1 420px", "minWidth": "340px", "border": "1px solid #e5e5e5", "borderRadius": "10px", "padding": "8px"}
filter_card_style = {"border": "1px solid #e5e5e5", "borderRadius": "10px", "padding": "10px", "marginBottom": "12px"}


app.layout = html.Div(
    [
        html.H1("Student Productivity Dashboard", style={"marginBottom": "8px"}),
        # html.P("Global filters are applied to all charts.", style={"marginTop": "0", "color": "#4f4f4f"}),
        dcc.Store(id="gender-last-valid", data=DEFAULT_GENDERS.copy()),
        html.Div(
            [
                html.H3("Global Filter", style={"margin": "0 0 10px 0"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Gender"),
                                dcc.Checklist(
                                    id="gender-filter",
                                    options=[{"label": value, "value": value} for value in GENDER_FILTER_OPTIONS],
                                    value=DEFAULT_GENDERS,
                                    inline=True,
                                    inputStyle={"marginRight": "6px"},
                                    labelStyle={"marginRight": "18px"},
                                ),
                            ],
                            style=control_style,
                        ),
                        html.Div(
                            [
                                html.Label("Age Range"),
                                dcc.RangeSlider(
                                    id="age-filter",
                                    min=DEFAULT_AGE_RANGE[0],
                                    max=DEFAULT_AGE_RANGE[1],
                                    step=1,
                                    value=DEFAULT_AGE_RANGE,
                                    marks=slider_marks(DEFAULT_AGE_RANGE[0], DEFAULT_AGE_RANGE[1], digits=0),
                                    allowCross=False,
                                    updatemode="drag",
                                ),
                            ],
                            style=control_style,
                        ),
                        html.Div(
                            [
                                html.Label("Main Activity Time Range"),
                                dcc.RangeSlider(
                                    id="main-activity-filter",
                                    min=DEFAULT_MAIN_ACTIVITY_RANGE[0],
                                    max=DEFAULT_MAIN_ACTIVITY_RANGE[1],
                                    step=0.1,
                                    value=DEFAULT_MAIN_ACTIVITY_RANGE,
                                    marks=slider_marks(
                                        DEFAULT_MAIN_ACTIVITY_RANGE[0], DEFAULT_MAIN_ACTIVITY_RANGE[1], digits=1
                                    ),
                                    allowCross=False,
                                    updatemode="drag",
                                ),
                            ],
                            style=control_style,
                        ),
                        html.Div(
                            [
                                html.Label("Productivity Score Range"),
                                dcc.RangeSlider(
                                    id="productivity-filter",
                                    min=DEFAULT_PRODUCTIVITY_RANGE[0],
                                    max=DEFAULT_PRODUCTIVITY_RANGE[1],
                                    step=1,
                                    value=DEFAULT_PRODUCTIVITY_RANGE,
                                    marks=slider_marks(DEFAULT_PRODUCTIVITY_RANGE[0], DEFAULT_PRODUCTIVITY_RANGE[1], digits=0),
                                    allowCross=False,
                                    updatemode="drag",
                                ),
                            ],
                            style=control_style,
                        ),
                        html.Div(
                            [
                                html.Button("Reset Filters", id="reset-filters", n_clicks=0, style={"height": "38px"}),
                            ],
                            style={"display": "flex", "flexDirection": "column", "gap": "6px", "minWidth": "140px"},
                        ),
                    ],
                    style={"display": "flex", "flexWrap": "wrap", "gap": "14px"},
                ),
            ],
            style=filter_card_style,
        ),
        html.Div(id="record-count", style={"fontWeight": "600", "marginBottom": "14px"}),
        html.Div(
            [
                html.Div(dcc.Graph(id="ternary-productivity", config={"displaylogo": False}), style=top_card_style),
                html.Div(dcc.Graph(id="ternary-density", config={"displaylogo": False}), style=top_card_style),
            ],
            style=row_style,
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="study-productivity", config={"displaylogo": False}), style=middle_card_style),
                html.Div(dcc.Graph(id="sleep-productivity", config={"displaylogo": False}), style=middle_card_style),
                html.Div(dcc.Graph(id="phone-productivity", config={"displaylogo": False}), style=middle_card_style),
            ],
            style=row_style,
        ),
        html.Div(
            [
                html.Div(dcc.Graph(id="focus-distribution", config={"displaylogo": False}), style=bottom_card_style),
                html.Div(dcc.Graph(id="stress-distribution", config={"displaylogo": False}), style=bottom_card_style),
            ],
            style=row_style,
        ),
    ],
    style={"maxWidth": "1500px", "margin": "0 auto", "padding": "18px", "fontFamily": "Segoe UI, Arial, sans-serif"},
)


@app.callback(
    Output("gender-filter", "value"),
    Output("age-filter", "value"),
    Output("main-activity-filter", "value"),
    Output("productivity-filter", "value"),
    Output("gender-last-valid", "data"),
    Input("reset-filters", "n_clicks"),
    Input("gender-filter", "value"),
    State("gender-last-valid", "data"),
    prevent_initial_call=True,
)
def sync_filters(
    _n_clicks: int, genders: list[str] | None, last_valid_genders: list[str] | None
) -> tuple[object, object, object, object, object]:
    triggered = ctx.triggered_id
    if triggered == "reset-filters":
        defaults = DEFAULT_GENDERS.copy()
        return (
            defaults,
            DEFAULT_AGE_RANGE.copy(),
            DEFAULT_MAIN_ACTIVITY_RANGE.copy(),
            DEFAULT_PRODUCTIVITY_RANGE.copy(),
            defaults,
        )

    if triggered == "gender-filter":
        if genders:
            return no_update, no_update, no_update, no_update, genders

        fallback = last_valid_genders if last_valid_genders else DEFAULT_GENDERS.copy()
        return fallback, no_update, no_update, no_update, fallback

    return no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("age-filter", "marks"),
    Output("main-activity-filter", "marks"),
    Output("productivity-filter", "marks"),
    Input("age-filter", "value"),
    Input("main-activity-filter", "value"),
    Input("productivity-filter", "value"),
)
def update_slider_marks(
    age_range: list[float] | None,
    main_activity_range: list[float] | None,
    productivity_range: list[float] | None,
) -> tuple[dict[float, str], dict[float, str], dict[float, str]]:
    selected_age = age_range if age_range is not None else DEFAULT_AGE_RANGE
    selected_main_activity = (
        main_activity_range if main_activity_range is not None else DEFAULT_MAIN_ACTIVITY_RANGE
    )
    selected_productivity = (
        productivity_range if productivity_range is not None else DEFAULT_PRODUCTIVITY_RANGE
    )

    return (
        slider_marks_with_selection(
            DEFAULT_AGE_RANGE[0],
            DEFAULT_AGE_RANGE[1],
            selected_age,
            digits=0,
        ),
        slider_marks_with_selection(
            DEFAULT_MAIN_ACTIVITY_RANGE[0],
            DEFAULT_MAIN_ACTIVITY_RANGE[1],
            selected_main_activity,
            digits=1,
        ),
        slider_marks_with_selection(
            DEFAULT_PRODUCTIVITY_RANGE[0],
            DEFAULT_PRODUCTIVITY_RANGE[1],
            selected_productivity,
            digits=0,
        ),
    )


@app.callback(
    Output("record-count", "children"),
    Output("ternary-productivity", "figure"),
    Output("ternary-density", "figure"),
    Output("study-productivity", "figure"),
    Output("phone-productivity", "figure"),
    Output("sleep-productivity", "figure"),
    Output("focus-distribution", "figure"),
    Output("stress-distribution", "figure"),
    Input("gender-filter", "value"),
    Input("age-filter", "value"),
    Input("main-activity-filter", "value"),
    Input("productivity-filter", "value"),
)
def update_dashboard(
    genders: list[str],
    age_range: list[float],
    main_activity_range: list[float],
    productivity_range: list[float],
):
    selected_genders = genders if genders is not None else []
    selected_age = age_range if age_range is not None else DEFAULT_AGE_RANGE
    selected_main_activity = (
        main_activity_range if main_activity_range is not None else DEFAULT_MAIN_ACTIVITY_RANGE
    )
    selected_productivity = (
        productivity_range if productivity_range is not None else DEFAULT_PRODUCTIVITY_RANGE
    )

    filtered = apply_filters(
        DATA_DF,
        selected_genders,
        selected_age,
        selected_main_activity,
        selected_productivity,
    )
    count_text = f"Records after filter: {len(filtered):,} / {TOTAL_RECORDS:,}"

    fig_ternary_productivity = build_ternary_productivity(filtered)
    fig_ternary_density = build_ternary_density(filtered)
    fig_study = build_density_panel(filtered, "study_hours_per_day", "Study Time")
    fig_phone = build_density_panel(filtered, "phone_usage_hours", "Phone Time")
    fig_sleep = build_density_panel(filtered, "sleep_hours", "Sleep Time")
    fig_focus = build_focus_distribution(filtered)
    fig_stress = build_stress_distribution(filtered)

    return (
        count_text,
        fig_ternary_productivity,
        fig_ternary_density,
        fig_study,
        fig_phone,
        fig_sleep,
        fig_focus,
        fig_stress,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
