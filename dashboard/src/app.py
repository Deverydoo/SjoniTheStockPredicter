"""
ArgusTrader Dashboard - Main Application
=========================================
Real-time trading dashboard using Dash and Plotly.
"""

import os
import random
from datetime import datetime, timedelta
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .components.charts import create_candlestick_chart, create_pnl_chart
from .components.cards import create_stats_card, create_position_card, create_signal_card, create_market_status_card


def create_app() -> Dash:
    """Create and configure the Dash application."""

    # Initialize Dash app with Bootstrap theme
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        update_title=None,
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
    )

    app.title = "ArgusTrader"

    # Custom CSS for dark theme
    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #0d1117;
                color: #c9d1d9;
            }
            .card {
                background-color: #161b22 !important;
                border-color: #30363d !important;
            }
            .navbar {
                background-color: #161b22 !important;
                border-bottom: 1px solid #30363d;
            }
            .table {
                color: #c9d1d9;
            }
            .table-hover tbody tr:hover {
                background-color: #21262d;
            }
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #0d1117;
            }
            ::-webkit-scrollbar-thumb {
                background: #30363d;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #484f58;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

    # Define layout
    app.layout = create_layout()

    # Register callbacks
    register_callbacks(app)

    return app


def create_layout() -> dbc.Container:
    """Create the main dashboard layout."""

    return dbc.Container(
        [
            # Header row with portfolio stats
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("ArgusTrader", className="text-primary mb-0"),
                            html.Small(
                                "NASDAQ Trading Engine",
                                className="text-muted",
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        html.Div(id="stat-portfolio"),
                        width=2,
                    ),
                    dbc.Col(
                        html.Div(id="stat-pnl"),
                        width=2,
                    ),
                    dbc.Col(
                        html.Div(id="stat-winrate"),
                        width=2,
                    ),
                    dbc.Col(
                        html.Div(id="stat-signals"),
                        width=2,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Span(id="connection-status"),
                            ],
                            className="text-end pt-3",
                        ),
                        width=1,
                    ),
                ],
                className="mb-4 align-items-center",
            ),
            # Main content row
            dbc.Row(
                [
                    # Left column: Rankings
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Span("Top 20 Rankings"),
                                        dbc.Badge(
                                            "Live",
                                            color="success",
                                            className="ms-2",
                                        ),
                                    ]
                                ),
                                dbc.CardBody(
                                    html.Div(
                                        id="rankings-table",
                                        children=create_rankings_table(),
                                    ),
                                    style={"maxHeight": "600px", "overflowY": "auto"},
                                ),
                            ]
                        ),
                        width=3,
                    ),
                    # Center column: Charts
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Span(
                                                id="selected-symbol-display",
                                                children="NVDA",
                                                className="h5 mb-0",
                                            ),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        "1m", id="btn-1m", size="sm", outline=True, color="primary"
                                                    ),
                                                    dbc.Button(
                                                        "5m", id="btn-5m", size="sm", outline=True, color="primary"
                                                    ),
                                                    dbc.Button(
                                                        "15m",
                                                        id="btn-15m",
                                                        size="sm",
                                                        outline=True,
                                                        color="primary",
                                                        active=True,
                                                    ),
                                                ],
                                                className="float-end",
                                            ),
                                        ],
                                        className="d-flex justify-content-between align-items-center",
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="price-chart",
                                                config={"displayModeBar": False},
                                                style={"height": "400px"},
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("P&L Performance"),
                                    dbc.CardBody(
                                        dcc.Graph(
                                            id="pnl-chart",
                                            config={"displayModeBar": False},
                                            style={"height": "180px"},
                                        ),
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    # Right column: Signals and Positions
                    dbc.Col(
                        [
                            # Market Status
                            html.Div(id="market-status-card", className="mb-3"),
                            # Signal Analysis
                            dbc.Card(
                                [
                                    dbc.CardHeader("Signal Analysis"),
                                    dbc.CardBody(
                                        html.Div(id="signal-details"),
                                    ),
                                ],
                                className="mb-3",
                            ),
                            # Positions
                            dbc.Card(
                                [
                                    dbc.CardHeader("Open Positions"),
                                    dbc.CardBody(
                                        html.Div(
                                            id="positions-container",
                                            style={"maxHeight": "250px", "overflowY": "auto"},
                                        ),
                                    ),
                                ]
                            ),
                        ],
                        width=3,
                    ),
                ]
            ),
            # Alerts row
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Alert(
                            id="alerts-bar",
                            children="System ready. Generating demo data...",
                            color="info",
                            className="mt-3 mb-0",
                        )
                    )
                ]
            ),
            # Interval components for updates
            dcc.Interval(id="interval-fast", interval=1000, n_intervals=0),  # 1 second
            dcc.Interval(id="interval-slow", interval=5000, n_intervals=0),  # 5 seconds
            # Store for selected symbol
            dcc.Store(id="store-selected-symbol", data="NVDA"),
        ],
        fluid=True,
        className="p-4",
    )


def create_rankings_table() -> dbc.Table:
    """Create the stock rankings table with demo data."""
    symbols = [
        ("NVDA", 0.85, "BUY"),
        ("AAPL", 0.72, "BUY"),
        ("TSLA", -0.45, "SELL"),
        ("MSFT", 0.68, "BUY"),
        ("AMZN", 0.55, "HOLD"),
        ("GOOGL", 0.62, "BUY"),
        ("META", -0.35, "SELL"),
        ("AMD", 0.78, "BUY"),
        ("NFLX", 0.42, "HOLD"),
        ("AVGO", 0.58, "BUY"),
    ]

    rows = []
    for i, (symbol, score, signal) in enumerate(symbols):
        signal_color = "#3fb950" if signal == "BUY" else "#f85149" if signal == "SELL" else "#d29922"
        rows.append(
            html.Tr(
                [
                    html.Td(str(i + 1)),
                    html.Td(symbol, className="fw-bold"),
                    html.Td(
                        html.Span(
                            signal,
                            className="badge",
                            style={"backgroundColor": signal_color},
                        )
                    ),
                    html.Td(f"{score:.2f}"),
                ],
                id={"type": "ranking-row", "symbol": symbol},
                style={"cursor": "pointer"},
            )
        )

    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th("#"), html.Th("Symbol"), html.Th("Signal"), html.Th("Score")])),
            html.Tbody(rows),
        ],
        striped=True,
        hover=True,
        size="sm",
    )


def generate_demo_bars(symbol: str, n_bars: int = 100) -> pd.DataFrame:
    """Generate demo OHLCV data for visualization."""
    now = datetime.now()
    base_prices = {
        "NVDA": 480.0, "AAPL": 175.0, "TSLA": 240.0, "MSFT": 380.0,
        "AMZN": 155.0, "GOOGL": 140.0, "META": 350.0, "AMD": 120.0,
        "NFLX": 450.0, "AVGO": 900.0,
    }
    base_price = base_prices.get(symbol, 100.0)

    data = []
    price = base_price

    for i in range(n_bars):
        timestamp = now - timedelta(minutes=n_bars - i)
        change = random.uniform(-1.0, 1.0)
        price = max(price + change, base_price * 0.95)
        price = min(price, base_price * 1.05)

        o = price
        h = price + random.uniform(0, 0.5)
        l = price - random.uniform(0, 0.5)
        c = price + random.uniform(-0.3, 0.3)
        v = random.randint(50000, 200000)

        data.append({
            "timestamp": timestamp,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "vwap": (o + h + l + c) / 4,
        })

    return pd.DataFrame(data)


def register_callbacks(app: Dash) -> None:
    """Register all dashboard callbacks."""

    @app.callback(
        Output("price-chart", "figure"),
        Output("selected-symbol-display", "children"),
        Input("interval-fast", "n_intervals"),
        Input("store-selected-symbol", "data"),
    )
    def update_price_chart(n_intervals: int, symbol: str) -> tuple:
        """Update the price chart with demo data."""
        df = generate_demo_bars(symbol)
        fig = create_candlestick_chart(df, symbol, height=380)
        return fig, symbol

    @app.callback(
        Output("pnl-chart", "figure"),
        Input("interval-fast", "n_intervals"),
    )
    def update_pnl_chart(n_intervals: int) -> go.Figure:
        """Update the P&L chart."""
        now = datetime.now()
        timestamps = [now - timedelta(minutes=i) for i in range(60, 0, -1)]

        # Generate cumulative P&L
        pnl = 0
        pnl_values = []
        for _ in timestamps:
            pnl += random.uniform(-50, 75)
            pnl_values.append(pnl)

        return create_pnl_chart(timestamps, pnl_values, height=160)

    @app.callback(
        Output("stat-portfolio", "children"),
        Output("stat-pnl", "children"),
        Output("stat-winrate", "children"),
        Output("stat-signals", "children"),
        Input("interval-slow", "n_intervals"),
    )
    def update_stats(n_intervals: int) -> tuple:
        """Update portfolio statistics."""
        portfolio = create_stats_card("Portfolio Value", "$125,432.50", color="primary")
        pnl = create_stats_card("Today's P&L", "+$1,234.56", subtitle="+0.98%", color="success")
        winrate = create_stats_card("Win Rate", "68.5%", subtitle="Last 30 days", color="info")
        signals = create_stats_card("Active Signals", "12", subtitle="3 BUY, 2 SELL", color="warning")
        return portfolio, pnl, winrate, signals

    @app.callback(
        Output("market-status-card", "children"),
        Input("interval-slow", "n_intervals"),
    )
    def update_market_status(n_intervals: int) -> Any:
        """Update market status indicator."""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()

        if weekday >= 5:
            status = "CLOSED"
            next_event = "Opens Monday 9:30 AM ET"
        elif hour < 9 or (hour == 9 and now.minute < 30):
            status = "PRE_MARKET"
            next_event = "Opens at 9:30 AM ET"
        elif hour >= 16:
            status = "AFTER_HOURS"
            next_event = "Closed for the day"
        else:
            status = "OPEN"
            next_event = "Closes at 4:00 PM ET"

        return create_market_status_card(status, next_event)

    @app.callback(
        Output("signal-details", "children"),
        Input("interval-fast", "n_intervals"),
        Input("store-selected-symbol", "data"),
    )
    def update_signal_details(n_intervals: int, symbol: str) -> Any:
        """Update signal analysis for selected symbol."""
        # Demo signals
        tech_score = random.uniform(0.5, 0.9)
        ml_score = random.uniform(0.4, 0.95)
        sent_score = random.uniform(0.3, 0.85)
        composite = (tech_score * 0.4 + ml_score * 0.4 + sent_score * 0.2)

        signal_type = "BUY" if composite > 0.6 else "SELL" if composite < 0.4 else "HOLD"
        signal_color = "#3fb950" if signal_type == "BUY" else "#f85149" if signal_type == "SELL" else "#d29922"

        return html.Div([
            html.Div([
                html.Label("Technical", className="small text-muted"),
                dbc.Progress(value=int(tech_score * 100), color="info", className="mb-2", style={"height": "8px"}),
            ]),
            html.Div([
                html.Label("ML Prediction", className="small text-muted"),
                dbc.Progress(value=int(ml_score * 100), color="info", className="mb-2", style={"height": "8px"}),
            ]),
            html.Div([
                html.Label("Sentiment", className="small text-muted"),
                dbc.Progress(value=int(sent_score * 100), color="info", className="mb-2", style={"height": "8px"}),
            ]),
            html.Hr(style={"borderColor": "#30363d"}),
            html.Div([
                html.Label("Composite Score", className="small text-muted"),
                dbc.Progress(
                    value=int(composite * 100),
                    color="success" if signal_type == "BUY" else "danger" if signal_type == "SELL" else "warning",
                    className="mb-3",
                    style={"height": "12px"},
                ),
            ]),
            html.Div([
                html.Span(
                    signal_type,
                    className="badge me-2",
                    style={"backgroundColor": signal_color, "fontSize": "1rem"},
                ),
                html.Span(f"{composite:.1%} confidence", className="text-muted"),
            ], className="mb-3"),
            html.Table([
                html.Tr([html.Td("Entry:", className="text-muted pe-3"), html.Td("$482.50")]),
                html.Tr([html.Td("Stop Loss:", className="text-muted pe-3"), html.Td("$475.00")]),
                html.Tr([html.Td("Take Profit:", className="text-muted pe-3"), html.Td("$510.00")]),
                html.Tr([html.Td("Risk/Reward:", className="text-muted pe-3"), html.Td("1:3.7")]),
            ], className="table table-sm mb-0"),
        ])

    @app.callback(
        Output("positions-container", "children"),
        Input("interval-slow", "n_intervals"),
    )
    def update_positions(n_intervals: int) -> Any:
        """Update open positions display."""
        positions = [
            {"symbol": "NVDA", "quantity": 50, "avg_cost": 465.00, "current_price": 482.15, "pnl": 857.50},
            {"symbol": "AAPL", "quantity": 100, "avg_cost": 172.50, "current_price": 175.23, "pnl": 273.00},
            {"symbol": "AMD", "quantity": 75, "avg_cost": 125.00, "current_price": 118.50, "pnl": -487.50},
        ]

        if not positions:
            return html.P("No open positions", className="text-muted text-center")

        return [create_position_card(**pos) for pos in positions]

    @app.callback(
        Output("connection-status", "children"),
        Input("interval-slow", "n_intervals"),
    )
    def update_connection_status(n_intervals: int) -> Any:
        """Update connection status indicator."""
        # Demo: always show connected
        return html.Span([
            html.Span("", style={
                "display": "inline-block",
                "width": "8px",
                "height": "8px",
                "borderRadius": "50%",
                "backgroundColor": "#3fb950",
                "marginRight": "6px",
            }),
            "Demo Mode",
        ], className="badge bg-success")


def main() -> None:
    """Main entry point for the dashboard."""

    # Get configuration from environment
    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("DASH_PORT", "8050"))
    debug = os.getenv("DASH_DEBUG", "true").lower() == "true"

    print(
        f"""
    ============================================================
                      ArgusTrader Dashboard
    ============================================================
      Running on: http://{host}:{port}
      Debug mode: {debug}
    ============================================================
    """
    )

    # Create and run app
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
