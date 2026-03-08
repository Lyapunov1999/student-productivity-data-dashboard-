from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
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


def slider_marks(min_value: float, max_value: float, digits: int = 0) -> dict[float, str]:
    mid = round((min_value + max_value) / 2, digits)
    return {
        round(min_value, digits): f"{min_value:.{digits}f}",
        mid: f"{mid:.{digits}f}",
        round(max_value, digits): f"{max_value:.{digits}f}",
    }


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


def build_ternary_productivity(df: pd.DataFrame) -> go.Figure:
    title = "Ternary Composition (Color = Productivity Score)"
    if df.empty:
        return empty_figure(title)

    fig = go.Figure(
        data=[
            go.Scatterternary(
                mode="markers",
                a=df["sleep_ratio"],
                b=df["study_ratio"],
                c=df["phone_ratio"],
                marker=dict(
                    size=6,
                    opacity=0.8,
                    color=df["productivity_score"],
                    colorscale="Viridis",
                    colorbar=dict(title="Productivity"),
                ),
                hovertemplate=(
                    "Sleep ratio: %{a:.2f}<br>"
                    "Study ratio: %{b:.2f}<br>"
                    "Phone ratio: %{c:.2f}<br>"
                    "Productivity: %{marker.color:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=40, r=20, t=55, b=40),
        ternary=dict(
            sum=1,
            aaxis=dict(title="Sleep Ratio", showgrid=False),
            baxis=dict(title="Study Ratio", showgrid=False),
            caxis=dict(title="Phone Ratio", showgrid=False),
        ),
    )
    return fig


def build_ternary_density(df: pd.DataFrame) -> go.Figure:
    title = "Ternary Density Contour"
    if df.empty:
        return empty_figure(title)

    if len(df) < 10:
        # Fallback for very small samples where contour density is unstable.
        fig = go.Figure(
            data=[
                go.Scatterternary(
                    mode="markers",
                    a=df["sleep_ratio"],
                    b=df["study_ratio"],
                    c=df["phone_ratio"],
                    marker=dict(size=6, color="#1f77b4", opacity=0.8),
                    showlegend=False,
                )
            ]
        )
    else:
        coordinates = np.vstack(
            [df["sleep_ratio"].to_numpy(), df["study_ratio"].to_numpy(), df["phone_ratio"].to_numpy()]
        )
        try:
            fig = ff.create_ternary_contour(
                coordinates=coordinates,
                pole_labels=["Sleep Ratio", "Study Ratio", "Phone Ratio"],
                ncontours=20,
                colorscale="Blues",
                interp_mode="cartesian",
                showscale=True,
            )
        except Exception:
            fig = go.Figure(
                data=[
                    go.Scatterternary(
                        mode="markers",
                        a=df["sleep_ratio"],
                        b=df["study_ratio"],
                        c=df["phone_ratio"],
                        marker=dict(size=6, color="#1f77b4", opacity=0.8),
                        showlegend=False,
                    )
                ]
            )

    for trace in fig.data:
        if hasattr(trace, "colorbar") and trace.colorbar:
            trace.colorbar.title = "Density"

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=40, r=20, t=55, b=40),
        ternary=dict(
            sum=1,
            aaxis=dict(title="Sleep Ratio", showgrid=False),
            baxis=dict(title="Study Ratio", showgrid=False),
            caxis=dict(title="Phone Ratio", showgrid=False),
        ),
    )
    return fig


def build_density_panel(df: pd.DataFrame, x_col: str, x_label: str) -> go.Figure:
    title = f"{x_label} vs Productivity"
    if df.empty:
        return empty_figure(title)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram2dContour(
            x=df[x_col],
            y=df["productivity_score"],
            colorscale="YlOrRd",
            ncontours=18,
            contours=dict(coloring="fill", showlabels=False),
            colorbar=dict(title="Density"),
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>Productivity: %{{y:.2f}}<extra></extra>",
            showscale=True,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=df[x_col],
            y=df["productivity_score"],
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
    title = "Focus Score Distribution"
    if df.empty:
        return empty_figure(title)

    values = df["focus_score"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            histnorm="probability density",
            nbinsx=30,
            name="Histogram",
            opacity=0.65,
            marker=dict(color="#4c78a8"),
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
                line=dict(color="#f58518", width=2),
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
    title = "Stress Level Distribution"
    if df.empty:
        return empty_figure(title)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["stress_level"],
            xbins=dict(start=0.5, end=10.5, size=1),
            marker=dict(color="#e45756"),
            name="Count",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=45, r=20, t=55, b=45),
        xaxis_title="Stress Level",
        yaxis_title="Count",
        bargap=0.08,
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1, range=[0.5, 10.5])
    return fig


DATA_DF = load_data(DATA_PATH)
TOTAL_RECORDS = len(DATA_DF)

GENDER_FILTER_OPTIONS = ["Male", "Female"]
DEFAULT_GENDERS = [g for g in GENDER_FILTER_OPTIONS if g in set(DATA_DF["gender"].dropna().unique())]
if not DEFAULT_GENDERS:
    DEFAULT_GENDERS = GENDER_FILTER_OPTIONS.copy()
DEFAULT_AGE_RANGE = [int(DATA_DF["age"].min()), int(DATA_DF["age"].max())]
DEFAULT_MAIN_ACTIVITY_RANGE = make_range(DATA_DF["main_activity_time"], round_digits=2)
DEFAULT_PRODUCTIVITY_RANGE = make_range(DATA_DF["productivity_score"], round_digits=0)


app = Dash(__name__)
app.title = "Student Productivity Dashboard"

control_style = {"display": "flex", "flexDirection": "column", "gap": "6px", "minWidth": "250px", "flex": "1 1 250px"}
row_style = {"display": "flex", "flexWrap": "wrap", "gap": "14px", "marginBottom": "16px"}
top_card_style = {"flex": "1 1 420px", "minWidth": "340px", "border": "1px solid #e5e5e5", "borderRadius": "10px", "padding": "8px"}
middle_card_style = {"flex": "1 1 320px", "minWidth": "280px", "border": "1px solid #e5e5e5", "borderRadius": "10px", "padding": "8px"}
bottom_card_style = {"flex": "1 1 420px", "minWidth": "340px", "border": "1px solid #e5e5e5", "borderRadius": "10px", "padding": "8px"}


app.layout = html.Div(
    [
        html.H1("Student Productivity Dashboard", style={"marginBottom": "8px"}),
        html.P("Global filters are applied to all charts.", style={"marginTop": "0", "color": "#4f4f4f"}),
        dcc.Store(id="gender-last-valid", data=DEFAULT_GENDERS.copy()),
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
                            tooltip={"placement": "bottom", "always_visible": False},
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
                            tooltip={"placement": "bottom", "always_visible": False},
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
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    style=control_style,
                ),
                html.Div(
                    [
                        html.Label("Actions"),
                        html.Button("Reset Filters", id="reset-filters", n_clicks=0, style={"height": "38px"}),
                    ],
                    style={"display": "flex", "flexDirection": "column", "gap": "6px", "minWidth": "140px"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "marginBottom": "12px"},
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
    app.run(debug=True)
