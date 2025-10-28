"""Risk management module for volatility, VaR, and drawdown analysis."""

from .volatility import (
    calculate_historical_volatility,
    fit_garch_model,
    detect_volatility_regimes,
)
from .var import (
    calculate_var,
    calculate_cvar,
    backtest_var,
)
from .drawdowns import (
    calculate_drawdowns,
    calculate_max_drawdown,
    underwater_plot,
)
from .metrics import (
    calculate_risk_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)

__all__ = [
    "calculate_historical_volatility",
    "fit_garch_model",
    "detect_volatility_regimes",
    "calculate_var",
    "calculate_cvar",
    "backtest_var",
    "calculate_drawdowns",
    "calculate_max_drawdown",
    "underwater_plot",
    "calculate_risk_metrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
]


