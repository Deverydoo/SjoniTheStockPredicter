"""
ArgusTrader Card Components
"""

from dash import html
import dash_bootstrap_components as dbc


def create_stats_card(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = "primary",
    icon: str = "",
) -> dbc.Card:
    """Create a statistics card."""

    color_map = {
        "primary": "#58a6ff",
        "success": "#3fb950",
        "danger": "#f85149",
        "warning": "#d29922",
        "info": "#8b949e",
    }

    accent_color = color_map.get(color, color_map["primary"])

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(icon, className="me-2") if icon else None,
                html.Small(title, className="text-muted"),
            ]),
            html.H3(value, className="mb-0", style={"color": accent_color}),
            html.Small(subtitle, className="text-muted") if subtitle else None,
        ]),
        className="h-100",
        style={
            "backgroundColor": "#161b22",
            "border": "1px solid #30363d",
            "borderLeft": f"4px solid {accent_color}",
        },
    )


def create_position_card(
    symbol: str,
    quantity: int,
    avg_cost: float,
    current_price: float,
    pnl: float,
) -> dbc.Card:
    """Create a position summary card."""

    pnl_color = "#3fb950" if pnl >= 0 else "#f85149"
    pnl_sign = "+" if pnl >= 0 else ""
    pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.H5(symbol, className="mb-0 d-inline"),
                html.Span(
                    f" {quantity} shares",
                    className="text-muted ms-2",
                ),
            ]),
            html.Hr(className="my-2", style={"borderColor": "#30363d"}),
            dbc.Row([
                dbc.Col([
                    html.Small("Avg Cost", className="text-muted d-block"),
                    html.Span(f"${avg_cost:.2f}"),
                ], width=6),
                dbc.Col([
                    html.Small("Current", className="text-muted d-block"),
                    html.Span(f"${current_price:.2f}"),
                ], width=6),
            ], className="mb-2"),
            html.Div([
                html.Span("P&L: ", className="text-muted"),
                html.Span(
                    f"{pnl_sign}${abs(pnl):.2f} ({pnl_sign}{pnl_pct:.1f}%)",
                    style={"color": pnl_color, "fontWeight": "bold"},
                ),
            ]),
        ]),
        className="mb-2",
        style={
            "backgroundColor": "#161b22",
            "border": "1px solid #30363d",
        },
    )


def create_signal_card(
    symbol: str,
    signal_type: str,
    strength: float,
    source: str,
    timestamp: str,
) -> dbc.Card:
    """Create a trading signal card."""

    signal_colors = {
        "BUY": "#3fb950",
        "SELL": "#f85149",
        "HOLD": "#d29922",
    }

    signal_color = signal_colors.get(signal_type, "#8b949e")
    strength_pct = int(strength * 100)

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(
                    signal_type,
                    className="badge me-2",
                    style={"backgroundColor": signal_color},
                ),
                html.Strong(symbol),
            ], className="mb-2"),
            dbc.Progress(
                value=strength_pct,
                color="success" if signal_type == "BUY" else "danger" if signal_type == "SELL" else "warning",
                className="mb-2",
                style={"height": "6px"},
            ),
            html.Div([
                html.Small(f"Source: {source}", className="text-muted me-3"),
                html.Small(timestamp, className="text-muted"),
            ]),
        ]),
        className="mb-2",
        style={
            "backgroundColor": "#161b22",
            "border": "1px solid #30363d",
        },
    )


def create_market_status_card(
    status: str,
    next_event: str = "",
) -> dbc.Card:
    """Create market status indicator card."""

    status_config = {
        "OPEN": {"color": "#3fb950", "text": "Market Open"},
        "CLOSED": {"color": "#f85149", "text": "Market Closed"},
        "PRE_MARKET": {"color": "#d29922", "text": "Pre-Market"},
        "AFTER_HOURS": {"color": "#d29922", "text": "After Hours"},
    }

    config = status_config.get(status, {"color": "#8b949e", "text": status})

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(
                    "",
                    className="status-indicator me-2",
                    style={
                        "display": "inline-block",
                        "width": "10px",
                        "height": "10px",
                        "borderRadius": "50%",
                        "backgroundColor": config["color"],
                    },
                ),
                html.Span(config["text"], style={"color": config["color"]}),
            ]),
            html.Small(next_event, className="text-muted") if next_event else None,
        ]),
        style={
            "backgroundColor": "#161b22",
            "border": "1px solid #30363d",
        },
    )
