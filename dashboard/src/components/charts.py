"""
ArgusTrader Chart Components
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Dark theme colors
COLORS = {
    "bg": "#0d1117",
    "card_bg": "#161b22",
    "text": "#c9d1d9",
    "text_muted": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "blue": "#58a6ff",
    "yellow": "#d29922",
    "grid": "#21262d",
    "border": "#30363d",
}


def create_candlestick_chart(
    df: pd.DataFrame,
    symbol: str = "",
    height: int = 400,
) -> go.Figure:
    """Create a candlestick chart with volume subplot."""

    if df.empty:
        # Return empty chart with placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for data...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color=COLORS["text_muted"]),
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor=COLORS["bg"],
            height=height,
        )
        return fig

    # Create subplots: candlestick on top, volume on bottom
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing=dict(line=dict(color=COLORS["green"]), fillcolor=COLORS["green"]),
            decreasing=dict(line=dict(color=COLORS["red"]), fillcolor=COLORS["red"]),
        ),
        row=1,
        col=1,
    )

    # Add VWAP line if available
    if "vwap" in df.columns and df["vwap"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["vwap"],
                name="VWAP",
                line=dict(color=COLORS["yellow"], width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )

    # Volume bars
    colors = [
        COLORS["green"] if close >= open_ else COLORS["red"]
        for close, open_ in zip(df["close"], df["open"])
    ]

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    # Update layout
    title = f"{symbol} - Real-time" if symbol else "Price Chart"
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"])),
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["bg"],
        height=height,
        margin=dict(l=50, r=20, t=40, b=20),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )

    # Update axes
    fig.update_xaxes(
        gridcolor=COLORS["grid"],
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=COLORS["grid"],
        showgrid=True,
        zeroline=False,
        title_text="Price ($)",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        gridcolor=COLORS["grid"],
        showgrid=True,
        zeroline=False,
        title_text="Volume",
        row=2,
        col=1,
    )

    return fig


def create_volume_chart(
    df: pd.DataFrame,
    height: int = 200,
) -> go.Figure:
    """Create a standalone volume chart."""

    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No volume data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_muted"]),
        )
    else:
        colors = [
            COLORS["green"] if close >= open_ else COLORS["red"]
            for close, open_ in zip(df["close"], df["open"])
        ]

        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                marker_color=colors,
                opacity=0.8,
            )
        )

    fig.update_layout(
        title="Volume",
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["bg"],
        height=height,
        margin=dict(l=50, r=20, t=40, b=20),
        showlegend=False,
    )

    fig.update_xaxes(gridcolor=COLORS["grid"], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], showgrid=True)

    return fig


def create_price_chart(
    df: pd.DataFrame,
    symbol: str = "",
    height: int = 300,
) -> go.Figure:
    """Create a simple line chart for price."""

    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="Waiting for data...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_muted"]),
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["close"],
                mode="lines",
                name="Price",
                line=dict(color=COLORS["blue"], width=2),
                fill="tozeroy",
                fillcolor="rgba(88, 166, 255, 0.1)",
            )
        )

    title = f"{symbol} Price" if symbol else "Price"
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"])),
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["bg"],
        height=height,
        margin=dict(l=50, r=20, t=40, b=20),
        showlegend=False,
    )

    fig.update_xaxes(gridcolor=COLORS["grid"], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], showgrid=True, title_text="Price ($)")

    return fig


def create_pnl_chart(
    timestamps: list,
    pnl_values: list,
    height: int = 250,
) -> go.Figure:
    """Create P&L over time chart."""

    fig = go.Figure()

    if not timestamps or not pnl_values:
        fig.add_annotation(
            text="No P&L data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS["text_muted"]),
        )
    else:
        colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in pnl_values]

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=pnl_values,
                mode="lines+markers",
                name="P&L",
                line=dict(color=COLORS["blue"], width=2),
                marker=dict(color=colors, size=6),
            )
        )

        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=COLORS["text_muted"],
            opacity=0.5,
        )

    fig.update_layout(
        title="Realized P&L",
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["bg"],
        height=height,
        margin=dict(l=50, r=20, t=40, b=20),
        showlegend=False,
    )

    fig.update_xaxes(gridcolor=COLORS["grid"], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], showgrid=True, title_text="P&L ($)")

    return fig
