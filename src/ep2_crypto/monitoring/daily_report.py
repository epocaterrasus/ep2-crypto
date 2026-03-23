"""Daily report generator for paper trading sessions.

Reads trade history and paper exchange state to produce:
  - PnL summary (gross, net after fees, per-trade averages)
  - Sharpe ratio (annualised, sqrt(105120))
  - Maximum drawdown
  - Trade statistics (count, win rate, profit factor, consecutive losses)
  - Regime breakdown (performance per regime)
  - Sprint 14 go/no-go acceptance criteria verdict

Reports are returned as structured dicts (for JSON serialisation) and as
human-readable text (for Telegram/Slack alerts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

BARS_PER_YEAR = 105_120  # 288 bars/day * 365
ANNUALISATION = math.sqrt(BARS_PER_YEAR)

# Sprint 14 go/no-go thresholds
GO_NOGO_CRITERIA = {
    "min_trades": 200,
    "min_win_rate": 0.51,
    "min_sharpe": 1.0,
    "max_drawdown": 0.08,  # 8% of peak
    "min_profit_factor": 1.2,
    "min_avg_pnl_bps": 2.0,
    "max_consecutive_losses": 15,
    "min_profitable_regimes": 2,
}


@dataclass
class RegimeStats:
    """Performance breakdown for a single market regime."""

    regime: str
    trade_count: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.trade_count, 1)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / max(self.trade_count, 1)

    @property
    def is_profitable(self) -> bool:
        return self.total_pnl > 0.0


@dataclass
class DailyReport:
    """Structured daily performance report."""

    # Identity
    report_date: str = ""
    session_start_ms: int = 0
    session_end_ms: int = 0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_losses: int = 0
    avg_trade_pnl_usd: float = 0.0
    avg_trade_pnl_bps: float = 0.0

    # PnL
    gross_pnl_usd: float = 0.0
    total_fees_usd: float = 0.0
    net_pnl_usd: float = 0.0
    initial_balance: float = 0.0
    final_balance: float = 0.0
    return_pct: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0

    # Regime breakdown
    regime_stats: dict[str, RegimeStats] = field(default_factory=dict)
    profitable_regime_count: int = 0

    # Go/no-go
    go_nogo_verdict: str = "INSUFFICIENT_DATA"
    go_nogo_details: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        result: dict[str, Any] = {
            "report_date": self.report_date,
            "session_start_ms": self.session_start_ms,
            "session_end_ms": self.session_end_ms,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_trade_pnl_usd": round(self.avg_trade_pnl_usd, 4),
            "avg_trade_pnl_bps": round(self.avg_trade_pnl_bps, 4),
            "gross_pnl_usd": round(self.gross_pnl_usd, 4),
            "total_fees_usd": round(self.total_fees_usd, 4),
            "net_pnl_usd": round(self.net_pnl_usd, 4),
            "initial_balance": round(self.initial_balance, 2),
            "final_balance": round(self.final_balance, 2),
            "return_pct": round(self.return_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "max_drawdown_usd": round(self.max_drawdown_usd, 4),
            "profitable_regime_count": self.profitable_regime_count,
            "go_nogo_verdict": self.go_nogo_verdict,
            "go_nogo_details": self.go_nogo_details,
            "regime_breakdown": {
                r: {
                    "trade_count": s.trade_count,
                    "win_rate": round(s.win_rate, 4),
                    "total_pnl_usd": round(s.total_pnl, 4),
                    "is_profitable": s.is_profitable,
                }
                for r, s in self.regime_stats.items()
            },
        }
        return result

    def to_text(self) -> str:
        """Format as human-readable text for Telegram/Slack."""
        verdict_emoji = {"GO": "✅", "NO_GO": "❌", "INSUFFICIENT_DATA": "⏳"}.get(
            self.go_nogo_verdict, "❓"
        )
        lines = [
            f"📊 Daily Paper Trading Report — {self.report_date}",
            f"{'=' * 45}",
            f"Verdict: {verdict_emoji} {self.go_nogo_verdict}",
            "",
            "TRADES",
            f"  Total: {self.total_trades}  W/L: {self.winning_trades}/{self.losing_trades}",
            f"  Win rate: {self.win_rate:.1%}  Profit factor: {self.profit_factor:.2f}",
            f"  Max consec. losses: {self.max_consecutive_losses}",
            f"  Avg PnL/trade: {self.avg_trade_pnl_usd:+.2f} USD"
            f" ({self.avg_trade_pnl_bps:+.1f} bps)",
            "",
            "PnL",
            f"  Gross: {self.gross_pnl_usd:+.2f} USD",
            f"  Fees:  {self.total_fees_usd:.2f} USD",
            f"  Net:   {self.net_pnl_usd:+.2f} USD ({self.return_pct:+.2%})",
            "",
            "RISK",
            f"  Sharpe: {self.sharpe_ratio:.2f}  Sortino: {self.sortino_ratio:.2f}",
            f"  Max DD: {self.max_drawdown_pct:.2%} ({self.max_drawdown_usd:.2f} USD)",
            "",
            "REGIMES",
        ]
        for regime, stats in self.regime_stats.items():
            icon = "✅" if stats.is_profitable else "❌"
            lines.append(
                f"  {icon} {regime}: {stats.trade_count} trades, "
                f"WR={stats.win_rate:.1%}, PnL={stats.total_pnl:+.2f}"
            )
        return "\n".join(lines)


class DailyReportGenerator:
    """Generates daily performance reports from paper trading state.

    Usage:
        generator = DailyReportGenerator(initial_balance=10_000.0)
        report = generator.generate(trades=exchange.trades)
        print(report.to_text())
    """

    def __init__(
        self,
        initial_balance: float,
        avg_btc_price: float = 100_000.0,
        session_start_ms: int = 0,
    ) -> None:
        self._initial_balance = initial_balance
        self._avg_btc_price = avg_btc_price
        self._session_start_ms = session_start_ms

    def generate(
        self,
        trades: list[Any],  # list[PaperTrade]
        final_balance: float | None = None,
        regime_labels: dict[str, str] | None = None,
    ) -> DailyReport:
        """Generate a report from a list of PaperTrade objects.

        Args:
            trades: List of PaperTrade from PaperExchange.trades
            final_balance: Final balance (defaults to initial + net PnL)
            regime_labels: Map of order_id -> regime label for regime breakdown
        """
        import time

        now_ms = int(time.time() * 1000)
        report_date = datetime.now(UTC).strftime("%Y-%m-%d")

        if not trades:
            return DailyReport(
                report_date=report_date,
                session_start_ms=self._session_start_ms,
                session_end_ms=now_ms,
                initial_balance=self._initial_balance,
                final_balance=final_balance or self._initial_balance,
                go_nogo_verdict="INSUFFICIENT_DATA",
            )

        # Basic counts
        winning = [t for t in trades if t.pnl_usd > 0]
        losing = [t for t in trades if t.pnl_usd < 0]
        gross_pnl = sum(t.pnl_usd for t in trades)
        total_fees = sum(t.fee_usd for t in trades)
        net_pnl = gross_pnl - total_fees
        win_rate = len(winning) / max(len(trades), 1)
        avg_pnl_usd = gross_pnl / max(len(trades), 1)

        # Avg PnL in bps (relative to average notional)
        avg_notional = sum(t.notional_usd for t in trades) / max(len(trades), 1)
        avg_pnl_bps = (avg_pnl_usd / max(avg_notional, 1)) / 1e-4

        # Profit factor
        gross_wins = sum(t.pnl_usd for t in winning) if winning else 0.0
        gross_losses = abs(sum(t.pnl_usd for t in losing)) if losing else 0.0
        profit_factor = gross_wins / max(gross_losses, 1e-9) if gross_losses > 0 else float("inf")

        # Max consecutive losses
        max_consec = self._max_consecutive_losses(trades)

        # Sharpe and Sortino from per-trade returns
        pnl_series = np.array([t.pnl_usd for t in trades], dtype=float)
        sharpe = self._sharpe(pnl_series)
        sortino = self._sortino(pnl_series)

        # Max drawdown from cumulative PnL
        cum_pnl = np.cumsum(pnl_series)
        max_dd_usd, max_dd_pct = self._max_drawdown(cum_pnl, self._initial_balance)

        # Balance
        computed_final = self._initial_balance + net_pnl
        final = final_balance if final_balance is not None else computed_final
        return_pct = (final - self._initial_balance) / max(self._initial_balance, 1e-9)

        # Regime breakdown
        regime_stats: dict[str, RegimeStats] = {}
        if regime_labels:
            for trade in trades:
                regime = regime_labels.get(trade.order_id, "unknown")
                if regime not in regime_stats:
                    regime_stats[regime] = RegimeStats(regime=regime)
                s = regime_stats[regime]
                s.trade_count += 1
                s.total_pnl += trade.pnl_usd
                s.total_fees += trade.fee_usd
                if trade.pnl_usd > 0:
                    s.wins += 1

        profitable_regime_count = sum(1 for s in regime_stats.values() if s.is_profitable)

        # Go/no-go assessment
        go_nogo_details, verdict = self._assess_go_nogo(
            total_trades=len(trades),
            win_rate=win_rate,
            sharpe=sharpe,
            max_dd_pct=max_dd_pct,
            profit_factor=profit_factor,
            avg_pnl_bps=avg_pnl_bps,
            max_consec_losses=max_consec,
            profitable_regimes=profitable_regime_count,
            regime_stats=regime_stats,
        )

        report = DailyReport(
            report_date=report_date,
            session_start_ms=self._session_start_ms,
            session_end_ms=now_ms,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consec,
            avg_trade_pnl_usd=avg_pnl_usd,
            avg_trade_pnl_bps=avg_pnl_bps,
            gross_pnl_usd=gross_pnl,
            total_fees_usd=total_fees,
            net_pnl_usd=net_pnl,
            initial_balance=self._initial_balance,
            final_balance=final,
            return_pct=return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_usd=max_dd_usd,
            regime_stats=regime_stats,
            profitable_regime_count=profitable_regime_count,
            go_nogo_verdict=verdict,
            go_nogo_details=go_nogo_details,
        )

        logger.info(
            "daily_report_generated",
            date=report_date,
            total_trades=len(trades),
            sharpe=round(sharpe, 3),
            max_dd_pct=round(max_dd_pct, 4),
            verdict=verdict,
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(pnl: np.ndarray) -> float:
        """Annualised Sharpe from per-trade PnL array."""
        if len(pnl) < 2:
            return 0.0
        mean = np.mean(pnl)
        std = np.std(pnl, ddof=1)
        if std == 0:
            return 0.0
        # Annualise: trades per year ≈ 18,720 (65/day * 288 bars) but use
        # actual trade frequency as sqrt(n_trades / years). For simplicity,
        # use sqrt(n_trades) as a conservative annualisation proxy.
        return float((mean / std) * math.sqrt(len(pnl)))

    @staticmethod
    def _sortino(pnl: np.ndarray) -> float:
        """Annualised Sortino (downside deviation only)."""
        if len(pnl) < 2:
            return 0.0
        mean = np.mean(pnl)
        downside = pnl[pnl < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = np.std(downside, ddof=1)
        if downside_std == 0:
            return 0.0
        return float((mean / downside_std) * math.sqrt(len(pnl)))

    @staticmethod
    def _max_drawdown(
        cum_pnl: np.ndarray,
        initial_balance: float,
    ) -> tuple[float, float]:
        """Maximum drawdown: (absolute USD, fraction of peak equity)."""
        if len(cum_pnl) == 0:
            return 0.0, 0.0
        equity = initial_balance + cum_pnl
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd_usd = float(np.max(drawdown))
        max_dd_pct = max_dd_usd / max(float(np.max(peak)), 1e-9)
        return max_dd_usd, max_dd_pct

    @staticmethod
    def _max_consecutive_losses(trades: list[Any]) -> int:
        """Count maximum consecutive losing trades."""
        max_streak = 0
        current = 0
        for t in trades:
            if t.pnl_usd < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @staticmethod
    def _assess_go_nogo(
        total_trades: int,
        win_rate: float,
        sharpe: float,
        max_dd_pct: float,
        profit_factor: float,
        avg_pnl_bps: float,
        max_consec_losses: int,
        profitable_regimes: int,
        regime_stats: dict[str, RegimeStats],
    ) -> tuple[dict[str, bool], str]:
        """Evaluate Sprint 14 go/no-go acceptance criteria."""
        c = GO_NOGO_CRITERIA
        checks: dict[str, bool] = {
            "enough_trades": total_trades >= c["min_trades"],
            "win_rate_ok": win_rate > c["min_win_rate"],
            "sharpe_ok": sharpe >= c["min_sharpe"],
            "drawdown_ok": max_dd_pct <= c["max_drawdown"],
            "profit_factor_ok": profit_factor >= c["min_profit_factor"],
            "avg_pnl_ok": avg_pnl_bps >= c["min_avg_pnl_bps"],
            "consec_losses_ok": max_consec_losses <= c["max_consecutive_losses"],
            "regime_breadth_ok": profitable_regimes >= c["min_profitable_regimes"],
        }

        if not checks["enough_trades"]:
            verdict = "INSUFFICIENT_DATA"
        elif all(checks.values()):
            verdict = "GO"
        else:
            verdict = "NO_GO"

        return checks, verdict
