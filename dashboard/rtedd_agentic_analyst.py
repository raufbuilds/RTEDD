from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _number(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_mw(value: Any) -> str:
    value = _number(value)
    return "N/A" if value is None else f"{value:,.0f} MW"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    value = _number(value)
    return "N/A" if value is None else f"{value:+.{digits}f}%"


def _safe_pct(numerator: Any, denominator: Any) -> Optional[float]:
    n, d = _number(numerator), _number(denominator)
    if n is None or d is None or abs(d) < 1e-9:
        return None
    return (n / d) * 100


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make a copy and normalize Date to datetime when possible."""
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if "Hour" in out.columns:
        out["Hour"] = pd.to_numeric(out["Hour"], errors="coerce")
    return out


# ---------------------------------------------------------------------
# Conversation context
# ---------------------------------------------------------------------

@dataclass
class AgentContext:
    last_intent: str = "system_status"
    last_subject: str = "system"
    last_date: Optional[pd.Timestamp] = None
    last_hour: Optional[int] = None
    last_question: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)

    def remember(self, question: str, intent: str, subject: str,
                 date: Optional[pd.Timestamp] = None,
                 hour: Optional[int] = None) -> None:
        self.last_question = question
        self.last_intent = intent
        self.last_subject = subject
        if date is not None:
            self.last_date = pd.Timestamp(date)
        if hour is not None:
            self.last_hour = int(hour)

        self.turns.append({
            "question": question,
            "intent": intent,
            "subject": subject,
            "date": str(date) if date is not None else None,
            "hour": hour,
        })
        self.turns = self.turns[-20:]


# ---------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------

class IntentEngine:
    """
    Local rule + scoring intent engine.

    This is deliberately transparent: keywords are scored instead of hidden
    behind an external language model.
    """

    INTENTS = {
        "anomaly_investigation": {
            "keywords": [
                "anomaly", "anomalies", "alert", "abnormal", "unusual",
                "outlier", "flagged", "incident", "wrong", "problem"
            ],
            "subject": "anomaly",
        },
        "next_hour_forecast": {
            "keywords": [
                "next hour", "next-hour", "next forecast", "forecast next",
                "what will demand", "predict next"
            ],
            "subject": "forecast",
        },
        "forecast_analysis": {
            "keywords": [
                "forecast", "prediction", "predict", "prophet", "lightgbm",
                "ensemble", "p10", "p90", "model agreement", "models agree"
            ],
            "subject": "forecast",
        },
        "benchmark_analysis": {
            "keywords": [
                "benchmark", "baseline", "expected demand", "behaving normally",
                "normal today", "vs expected", "versus expected"
            ],
            "subject": "benchmark",
        },
        "live_model_analysis": {
            "keywords": [
                "live-trained", "live trained", "model performance",
                "which model", "best model", "closest model", "model accuracy"
            ],
            "subject": "live_models",
        },
        "demand_change": {
            "keywords": [
                "why is demand", "why demand", "increasing", "decreasing",
                "going up", "going down", "changed", "change", "rise", "drop",
                "higher", "lower", "trend"
            ],
            "subject": "demand",
        },
        "historical_analysis": {
            "keywords": [
                "yesterday", "last week", "history", "historical", "previous",
                "past", "average", "maximum", "minimum", "peak"
            ],
            "subject": "history",
        },
        "weather_analysis": {
            "keywords": [
                "weather", "temperature", "rain", "wind", "humidity", "heat",
                "cold"
            ],
            "subject": "weather",
        },
        "system_status": {
            "keywords": [
                "what's happening", "whats happening", "status", "system",
                "right now", "current", "overview", "summary"
            ],
            "subject": "system",
        },
    }

    FOLLOW_UP_WORDS = {
        "it", "that", "this", "those", "them", "what about", "and yesterday"
    }

    def detect(self, question: str, context: AgentContext) -> tuple[str, str, dict[str, float]]:
        q = question.lower().strip()
        scores: dict[str, float] = {}

        for intent, config in self.INTENTS.items():
            score = 0.0
            for keyword in config["keywords"]:
                if keyword in q:
                    # Longer phrases are more specific.
                    score += 1.0 + min(len(keyword.split()) * 0.35, 1.5)
            scores[intent] = score

        # Strong exact patterns.
        if re.search(r"\bwhy\b.*\b(anomaly|alert|flag)", q):
            scores["anomaly_investigation"] += 4
        if "next" in q and "hour" in q:
            scores["next_hour_forecast"] += 5
        if ("prophet" in q or "lightgbm" in q or "ensemble" in q) and "today" in q:
            scores["live_model_analysis"] += 2

        best_intent, best_score = max(scores.items(), key=lambda item: item[1])

        # Resolve vague follow-up questions using previous context.
        is_follow_up = len(q.split()) <= 8 and any(word in q for word in self.FOLLOW_UP_WORDS)
        if best_score == 0 or is_follow_up:
            if context.last_intent:
                best_intent = context.last_intent

        subject = self.INTENTS.get(best_intent, {}).get("subject", context.last_subject)
        return best_intent, subject, scores


# ---------------------------------------------------------------------
# Investigation planner
# ---------------------------------------------------------------------

class InvestigationPlanner:
    """Creates explicit multi-step plans from intent."""

    PLANS = {
        "system_status": [
            "system_snapshot",
            "current_models",
            "next_hour_forecast",
            "recent_anomalies",
        ],
        "demand_change": [
            "demand_change",
            "benchmark_comparison",
            "current_models",
            "recent_anomalies",
        ],
        "next_hour_forecast": [
            "next_hour_forecast",
            "current_models",
        ],
        "forecast_analysis": [
            "next_hour_forecast",
            "current_models",
            "live_model_comparison",
        ],
        "benchmark_analysis": [
            "benchmark_comparison",
            "demand_change",
        ],
        "live_model_analysis": [
            "live_model_comparison",
            "current_models",
        ],
        "anomaly_investigation": [
            "recent_anomalies",
            "demand_change",
            "benchmark_comparison",
            "current_models",
        ],
        "historical_analysis": [
            "historical_summary",
            "benchmark_comparison",
        ],
        "weather_analysis": [
            "weather_summary",
            "demand_change",
        ],
    }

    def create_plan(self, intent: str, question: str) -> list[str]:
        plan = list(self.PLANS.get(intent, ["system_snapshot"]))

        # Allow focused questions to request less noise.
        q = question.lower()
        if "only" in q and "forecast" in q:
            return ["next_hour_forecast"]

        return plan


# ---------------------------------------------------------------------
# Internal RTEDD analytical tools
# ---------------------------------------------------------------------

class RTEDDTools:
    """
    Approved internal tools.

    forecast_fn may be supplied from the existing RTEDD project.
    It should return a dataframe containing Hour and, where available:
        Prophet, LightGBM, Ensemble, Ensemble_P10, Ensemble_P90
    """

    def __init__(
        self,
        df: pd.DataFrame,
        forecast_fn: Optional[Callable[..., pd.DataFrame]] = None,
        weather_df: Optional[pd.DataFrame] = None,
    ):
        self.df = _normalize_columns(df)
        self.forecast_fn = forecast_fn
        self.weather_df = _normalize_columns(weather_df) if weather_df is not None else pd.DataFrame()

    def _latest_day(self) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
        if self.df.empty or "Date" not in self.df.columns:
            return pd.DataFrame(), None
        valid = self.df.dropna(subset=["Date"])
        if valid.empty:
            return pd.DataFrame(), None
        latest = valid["Date"].max()
        rows = valid[valid["Date"] == latest].sort_values("Hour")
        return rows, latest

    def _forecast_for_latest_day(self, include_target_date: bool = True) -> pd.DataFrame:
        if self.forecast_fn is None:
            return pd.DataFrame()

        rows, date = self._latest_day()
        if date is None:
            return pd.DataFrame()

        try:
            fc = self.forecast_fn(self.df.copy(), date, include_target_date=include_target_date)
        except TypeError:
            # Compatibility fallback for projects using a simpler function signature.
            try:
                fc = self.forecast_fn(self.df.copy(), date)
            except Exception:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

        return _normalize_columns(fc) if isinstance(fc, pd.DataFrame) else pd.DataFrame()

    def system_snapshot(self) -> dict[str, Any]:
        rows, date = self._latest_day()
        if rows.empty or date is None:
            return {"available": False, "reason": "No RTEDD data available."}

        latest = rows.iloc[-1]
        current = _number(latest.get("Ontario Demand"))
        expected = _number(latest.get("Expected Demand"))

        return {
            "available": True,
            "date": str(date.date()),
            "hour": int(latest["Hour"]) if pd.notna(latest.get("Hour")) else None,
            "current_demand_mw": current,
            "expected_demand_mw": expected,
            "records_available": int(len(self.df)),
            "latest_day_records": int(len(rows)),
            "timezone": "Ontario Time (ET)",
        }

    def demand_change(self) -> dict[str, Any]:
        rows, date = self._latest_day()
        if len(rows) < 2 or "Ontario Demand" not in rows.columns:
            return {"available": False, "reason": "Not enough demand observations."}

        latest = rows.iloc[-1]
        previous = rows.iloc[-2]

        current = _number(latest["Ontario Demand"])
        prev = _number(previous["Ontario Demand"])
        expected = _number(latest.get("Expected Demand"))

        change_mw = current - prev if current is not None and prev is not None else None
        change_pct = _safe_pct(change_mw, prev)
        baseline_gap_mw = current - expected if current is not None and expected is not None else None
        baseline_gap_pct = _safe_pct(baseline_gap_mw, expected)

        return {
            "available": True,
            "date": str(date.date()) if date is not None else None,
            "hour": int(latest["Hour"]),
            "previous_hour": int(previous["Hour"]),
            "current_demand_mw": current,
            "previous_demand_mw": prev,
            "change_mw": change_mw,
            "change_percent": change_pct,
            "expected_demand_mw": expected,
            "baseline_gap_mw": baseline_gap_mw,
            "baseline_gap_percent": baseline_gap_pct,
        }

    def next_hour_forecast(self) -> dict[str, Any]:
        rows, date = self._latest_day()
        if rows.empty:
            return {"available": False, "reason": "No current day data."}

        latest_hour = int(pd.to_numeric(rows["Hour"], errors="coerce").dropna().max())
        next_hour = latest_hour + 1
        # Ontario hourly datasets may use 1-24. Keep compatibility with either convention.
        if next_hour > 24:
            next_hour = 1

        fc = self._forecast_for_latest_day()
        if fc.empty or "Hour" not in fc.columns:
            return {
                "available": False,
                "next_hour": next_hour,
                "reason": "Forecast function unavailable or returned no forecast data.",
            }

        match = fc[pd.to_numeric(fc["Hour"], errors="coerce") == next_hour]
        if match.empty:
            return {
                "available": False,
                "next_hour": next_hour,
                "reason": "No forecast found for the next hour.",
            }

        row = match.iloc[0]
        result = {
            "available": True,
            "date": str(date.date()) if date is not None else None,
            "next_hour": next_hour,
            "prophet_mw": _number(row.get("Prophet")),
            "lightgbm_mw": _number(row.get("LightGBM")),
            "ensemble_mw": _number(row.get("Ensemble")),
            "ensemble_p10_mw": _number(row.get("Ensemble_P10")),
            "ensemble_p90_mw": _number(row.get("Ensemble_P90")),
        }

        p = result["prophet_mw"]
        l = result["lightgbm_mw"]
        if p is not None and l is not None:
            result["model_difference_mw"] = abs(p - l)
            result["model_difference_percent"] = _safe_pct(abs(p - l), (p + l) / 2)
        return result

    def current_models(self) -> dict[str, Any]:
        rows, date = self._latest_day()
        if rows.empty:
            return {"available": False, "reason": "No current day data."}

        latest = rows.iloc[-1]
        hour = int(latest["Hour"])
        actual = _number(latest.get("Ontario Demand"))

        fc = self._forecast_for_latest_day()
        result = {
            "available": True,
            "date": str(date.date()) if date is not None else None,
            "hour": hour,
            "actual_mw": actual,
        }

        if fc.empty or "Hour" not in fc.columns:
            result["forecast_available"] = False
            return result

        match = fc[pd.to_numeric(fc["Hour"], errors="coerce") == hour]
        if match.empty:
            result["forecast_available"] = False
            return result

        row = match.iloc[0]
        result["forecast_available"] = True

        for source, target in [
            ("Prophet", "prophet_mw"),
            ("LightGBM", "lightgbm_mw"),
            ("Ensemble", "ensemble_mw"),
        ]:
            prediction = _number(row.get(source))
            result[target] = prediction
            if actual is not None and prediction is not None:
                result[target.replace("_mw", "_error_mw")] = abs(actual - prediction)

        return result

    def benchmark_comparison(self) -> dict[str, Any]:
        """
        Today vs Forecast Benchmark.
        This intentionally does NOT evaluate live model accuracy.
        """
        rows, date = self._latest_day()
        required = {"Ontario Demand", "Expected Demand"}
        if rows.empty or not required.issubset(rows.columns):
            return {
                "available": False,
                "reason": "Actual or Expected Demand baseline is unavailable.",
            }

        d = rows.dropna(subset=["Ontario Demand", "Expected Demand"]).copy()
        if d.empty:
            return {"available": False, "reason": "No comparable benchmark records."}

        diff = d["Ontario Demand"].astype(float) - d["Expected Demand"].astype(float)
        latest = d.iloc[-1]

        return {
            "available": True,
            "concept": "Today vs Forecast Benchmark",
            "date": str(date.date()) if date is not None else None,
            "observations": int(len(d)),
            "mean_deviation_mw": float(diff.mean()),
            "mean_absolute_deviation_mw": float(diff.abs().mean()),
            "latest_actual_mw": _number(latest["Ontario Demand"]),
            "latest_expected_mw": _number(latest["Expected Demand"]),
            "latest_deviation_mw": float(diff.iloc[-1]),
            "latest_deviation_percent": _safe_pct(
                diff.iloc[-1], latest["Expected Demand"]
            ),
        }

    def live_model_comparison(self) -> dict[str, Any]:
        """
        Today vs Live-Trained Forecast.
        This intentionally remains separate from benchmark comparison.
        """
        rows, date = self._latest_day()
        if rows.empty or "Ontario Demand" not in rows.columns:
            return {"available": False, "reason": "Actual demand unavailable."}

        fc = self._forecast_for_latest_day()
        if fc.empty or "Hour" not in fc.columns:
            return {"available": False, "reason": "Live forecast unavailable."}

        actual = rows[["Hour", "Ontario Demand"]].copy()
        merged = actual.merge(fc, on="Hour", how="inner")

        if merged.empty:
            return {"available": False, "reason": "No matching live forecast hours."}

        result = {
            "available": True,
            "concept": "Today vs Live-Trained Forecast",
            "date": str(date.date()) if date is not None else None,
            "observations": int(len(merged)),
        }

        maes: dict[str, float] = {}
        for col, key in [
            ("Prophet", "prophet_mae_mw"),
            ("LightGBM", "lightgbm_mae_mw"),
            ("Ensemble", "ensemble_mae_mw"),
        ]:
            if col in merged.columns:
                valid = merged[["Ontario Demand", col]].dropna()
                if not valid.empty:
                    mae = float((valid["Ontario Demand"].astype(float) - valid[col].astype(float)).abs().mean())
                    result[key] = mae
                    maes[key] = mae

        if maes:
            best_key, _ = min(maes.items(), key=lambda item: item[1])
            result["best_model"] = {
                "prophet_mae_mw": "Prophet",
                "lightgbm_mae_mw": "LightGBM",
                "ensemble_mae_mw": "Ensemble",
            }.get(best_key, best_key)

        return result

    def recent_anomalies(self, limit: int = 5) -> dict[str, Any]:
        if self.df.empty or "Anomaly" not in self.df.columns:
            return {"available": False, "events": [], "reason": "Anomaly data unavailable."}

        d = self.df.copy()
        anomaly_values = d["Anomaly"]
        # Supports booleans, 0/1 and common string labels.
        mask = (
            anomaly_values.astype(str).str.lower().isin(["true", "1", "yes", "anomaly"])
            | anomaly_values.fillna(False).astype(bool)
        )
        events = d[mask].sort_values(["Date", "Hour"], ascending=False).head(max(1, min(limit, 20)))

        result = []
        for _, row in events.iterrows():
            actual = _number(row.get("Ontario Demand"))
            expected = _number(row.get("Expected Demand"))
            deviation = actual - expected if actual is not None and expected is not None else None
            result.append({
                "date": str(pd.Timestamp(row["Date"]).date()) if pd.notna(row.get("Date")) else None,
                "hour": int(row["Hour"]) if pd.notna(row.get("Hour")) else None,
                "actual_mw": actual,
                "expected_mw": expected,
                "deviation_mw": deviation,
                "deviation_percent": _safe_pct(deviation, expected),
                "anomaly_score": _number(row.get("Anomaly Score")),
            })

        return {
            "available": True,
            "count": int(len(events)),
            "events": result,
        }

    def historical_summary(self, days: int = 7) -> dict[str, Any]:
        if self.df.empty or "Date" not in self.df.columns or "Ontario Demand" not in self.df.columns:
            return {"available": False, "reason": "Historical demand data unavailable."}

        latest = self.df["Date"].max()
        if pd.isna(latest):
            return {"available": False, "reason": "Invalid dates."}

        days = max(1, min(int(days), 365))
        start = latest - pd.Timedelta(days=days - 1)
        d = self.df[self.df["Date"] >= start].dropna(subset=["Ontario Demand"])

        if d.empty:
            return {"available": False, "reason": "No records in selected period."}

        peak = d.loc[d["Ontario Demand"].idxmax()]
        low = d.loc[d["Ontario Demand"].idxmin()]

        return {
            "available": True,
            "days": days,
            "period_start": str(pd.Timestamp(d["Date"].min()).date()),
            "period_end": str(pd.Timestamp(d["Date"].max()).date()),
            "average_demand_mw": float(d["Ontario Demand"].mean()),
            "maximum_demand_mw": float(peak["Ontario Demand"]),
            "maximum_date": str(pd.Timestamp(peak["Date"]).date()),
            "maximum_hour": int(peak["Hour"]) if pd.notna(peak.get("Hour")) else None,
            "minimum_demand_mw": float(low["Ontario Demand"]),
            "minimum_date": str(pd.Timestamp(low["Date"]).date()),
            "minimum_hour": int(low["Hour"]) if pd.notna(low.get("Hour")) else None,
        }

    def weather_summary(self) -> dict[str, Any]:
        """
        Uses local weather data only if supplied.
        This tool does not invent causal weather impact.
        """
        if self.weather_df.empty:
            return {
                "available": False,
                "reason": "No local weather dataframe was supplied to the agent.",
            }

        latest = self.weather_df.iloc[-1].to_dict()
        allowed = [
            "Temperature", "Temp", "Humidity", "Wind Speed",
            "Weather", "Condition", "Date", "Hour"
        ]
        summary = {k: latest[k] for k in allowed if k in latest and pd.notna(latest[k])}
        return {"available": True, "weather": summary}


# ---------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------

class FindingEngine:
    """Turns raw evidence into ranked, structured findings."""

    def build(self, evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        findings = []

        change = evidence.get("demand_change", {})
        if change.get("available"):
            pct = _number(change.get("change_percent"))
            if pct is not None:
                if pct >= 5:
                    severity = "high"
                    statement = f"Demand increased sharply by {pct:.1f}% from the previous hour."
                elif pct <= -5:
                    severity = "high"
                    statement = f"Demand decreased sharply by {abs(pct):.1f}% from the previous hour."
                elif abs(pct) >= 2:
                    severity = "medium"
                    direction = "increased" if pct > 0 else "decreased"
                    statement = f"Demand {direction} by {abs(pct):.1f}% from the previous hour."
                else:
                    severity = "low"
                    statement = f"Demand changed by {pct:+.1f}% from the previous hour."

                findings.append({
                    "priority": {"high": 90, "medium": 65, "low": 30}[severity],
                    "severity": severity,
                    "title": "Demand movement",
                    "statement": statement,
                })

            baseline_pct = _number(change.get("baseline_gap_percent"))
            if baseline_pct is not None and abs(baseline_pct) >= 3:
                direction = "above" if baseline_pct > 0 else "below"
                findings.append({
                    "priority": 75 if abs(baseline_pct) >= 8 else 55,
                    "severity": "high" if abs(baseline_pct) >= 8 else "medium",
                    "title": "Baseline deviation",
                    "statement": f"Current demand is {abs(baseline_pct):.1f}% {direction} the expected baseline.",
                })

        forecast = evidence.get("next_hour_forecast", {})
        if forecast.get("available"):
            disagreement = _number(forecast.get("model_difference_percent"))
            if disagreement is not None:
                if disagreement <= 2:
                    label = "strong"
                elif disagreement <= 5:
                    label = "moderate"
                else:
                    label = "weak"

                findings.append({
                    "priority": 70 if label == "weak" else 45,
                    "severity": "medium" if label == "weak" else "low",
                    "title": "Model agreement",
                    "statement": (
                        f"Prophet and LightGBM show {label} agreement for the next-hour forecast "
                        f"({disagreement:.1f}% difference)."
                    ),
                })

        anomalies = evidence.get("recent_anomalies", {})
        if anomalies.get("available") and anomalies.get("count", 0) > 0:
            findings.append({
                "priority": 95,
                "severity": "high",
                "title": "Recent anomalies",
                "statement": f"The system found {anomalies['count']} recent anomaly event(s) in the latest investigation window.",
            })

        benchmark = evidence.get("benchmark_comparison", {})
        if benchmark.get("available"):
            pct = _number(benchmark.get("latest_deviation_percent"))
            if pct is not None and abs(pct) >= 3:
                direction = "above" if pct > 0 else "below"
                findings.append({
                    "priority": 72,
                    "severity": "medium",
                    "title": "Benchmark behavior",
                    "statement": f"Latest demand is {abs(pct):.1f}% {direction} the established benchmark.",
                })

        live = evidence.get("live_model_comparison", {})
        if live.get("available") and live.get("best_model"):
            findings.append({
                "priority": 40,
                "severity": "low",
                "title": "Live model performance",
                "statement": f"{live['best_model']} currently has the lowest available MAE for today's live-trained comparison.",
            })

        return sorted(findings, key=lambda x: x["priority"], reverse=True)


# ---------------------------------------------------------------------
# Explanation engine
# ---------------------------------------------------------------------

class ExplanationEngine:

    def respond(
        self,
        question: str,
        intent: str,
        evidence: dict[str, dict[str, Any]],
        findings: list[dict[str, Any]],
    ) -> str:
        if intent == "next_hour_forecast":
            return self._next_hour(evidence)

        if intent == "anomaly_investigation":
            return self._anomaly(evidence, findings)

        if intent == "benchmark_analysis":
            return self._benchmark(evidence, findings)

        if intent == "live_model_analysis":
            return self._live_models(evidence)

        if intent == "forecast_analysis":
            return self._forecast_analysis(evidence, findings)

        if intent == "historical_analysis":
            return self._historical(evidence)

        if intent == "weather_analysis":
            return self._weather(evidence)

        if intent == "demand_change":
            return self._demand_change(evidence, findings)

        return self._system(evidence, findings)

    def _next_hour(self, evidence):
        f = evidence.get("next_hour_forecast", {})
        if not f.get("available"):
            return f"## 🔮 Next-Hour Forecast\n\nI could not retrieve the next-hour forecast. **Reason:** {f.get('reason', 'Forecast data unavailable.')}"

        lines = [
            "## 🔮 Next-Hour Forecast",
            f"**Forecast hour:** {f.get('next_hour'):02d}:00 Ontario Time (ET)",
            "",
            f"⚡ **Ensemble:** {_fmt_mw(f.get('ensemble_mw'))}",
            f"🟠 Prophet: {_fmt_mw(f.get('prophet_mw'))}",
            f"🟢 LightGBM: {_fmt_mw(f.get('lightgbm_mw'))}",
        ]

        p10, p90 = f.get("ensemble_p10_mw"), f.get("ensemble_p90_mw")
        if _number(p10) is not None and _number(p90) is not None:
            lines += [
                "",
                f"**Expected range (P10–P90):** {_fmt_mw(p10)} — {_fmt_mw(p90)}",
            ]

        diff = _number(f.get("model_difference_percent"))
        if diff is not None:
            agreement = "strong" if diff <= 2 else "moderate" if diff <= 5 else "weak"
            lines += [
                "",
                f"**Model agreement:** {agreement.capitalize()} ({diff:.1f}% Prophet–LightGBM difference).",
            ]

        return "\n".join(lines)

    def _anomaly(self, evidence, findings):
        a = evidence.get("recent_anomalies", {})
        if not a.get("available") or not a.get("events"):
            return "## 🚨 Anomaly Investigation\n\nI found no recent anomaly events in the available RTEDD data."

        event = a["events"][0]
        lines = [
            "## 🚨 Latest Anomaly Investigation",
            "",
            "### What happened",
            f"At **{event.get('hour'):02d}:00 on {event.get('date')}**, demand was {_fmt_mw(event.get('actual_mw'))}.",
        ]

        if _number(event.get("expected_mw")) is not None:
            lines.append(f"The expected baseline was {_fmt_mw(event.get('expected_mw'))}.")

        if _number(event.get("deviation_percent")) is not None:
            direction = "above" if event["deviation_percent"] > 0 else "below"
            lines.append(
                f"That is **{abs(event['deviation_percent']):.1f}% {direction} the expected baseline**."
            )

        if _number(event.get("anomaly_score")) is not None:
            lines.append(f"Recorded anomaly score: **{event['anomaly_score']:.3f}**.")

        lines += [
            "",
            "### Evidence-based interpretation",
            "The anomaly indicates that observed demand deviated materially from the RTEDD expected pattern.",
        ]

        change = evidence.get("demand_change", {})
        if change.get("available") and _number(change.get("change_percent")) is not None:
            lines.append(
                f"The latest hourly movement is {_fmt_pct(change.get('change_percent'))}, "
                "which provides additional context but does not by itself prove the cause of the anomaly."
            )

        lines += [
            "",
            "### Important",
            "RTEDD can identify the deviation and supporting evidence. It should not claim a specific external cause unless supporting data establishes that cause.",
        ]
        return "\n".join(lines)

    def _benchmark(self, evidence, findings):
        b = evidence.get("benchmark_comparison", {})
        if not b.get("available"):
            return f"## 📉 Today vs Forecast Benchmark\n\n{b.get('reason', 'Benchmark data is unavailable.')}"

        lines = [
            "## 📉 Today vs Forecast Benchmark",
            "",
            "*This view answers whether today's observed demand is behaving relative to the established expected baseline.*",
            "",
            f"Latest actual demand: **{_fmt_mw(b.get('latest_actual_mw'))}**",
            f"Latest expected benchmark: **{_fmt_mw(b.get('latest_expected_mw'))}**",
            f"Latest deviation: **{_fmt_pct(b.get('latest_deviation_percent'))}**",
            f"Mean absolute deviation today: **{_fmt_mw(b.get('mean_absolute_deviation_mw'))}**",
        ]
        return "\n".join(lines)

    def _live_models(self, evidence):
        m = evidence.get("live_model_comparison", {})
        if not m.get("available"):
            return f"## 🧠 Today vs Live-Trained Forecast\n\n{m.get('reason', 'Live model comparison unavailable.')}"

        lines = [
            "## 🧠 Today vs Live-Trained Forecast",
            "",
            "*This is separate from the Forecast Benchmark. It evaluates live Prophet, LightGBM and Ensemble forecasts against today's observed demand.*",
            "",
        ]

        values = [
            ("Prophet MAE", m.get("prophet_mae_mw")),
            ("LightGBM MAE", m.get("lightgbm_mae_mw")),
            ("Ensemble MAE", m.get("ensemble_mae_mw")),
        ]
        for name, value in values:
            if _number(value) is not None:
                lines.append(f"**{name}:** {_fmt_mw(value)}")

        if m.get("best_model"):
            lines += ["", f"**Best available live performance:** {m['best_model']}"]
        return "\n".join(lines)

    def _forecast_analysis(self, evidence, findings):
        parts = [self._next_hour(evidence), "", "---", "", self._live_models(evidence)]
        return "\n".join(parts)

    def _historical(self, evidence):
        h = evidence.get("historical_summary", {})
        if not h.get("available"):
            return f"## 📊 Historical Analysis\n\n{h.get('reason', 'Historical data unavailable.')}"

        return "\n".join([
            f"## 📊 Historical Demand — Last {h.get('days')} Days",
            "",
            f"**Period:** {h.get('period_start')} to {h.get('period_end')}",
            f"Average demand: **{_fmt_mw(h.get('average_demand_mw'))}**",
            f"Peak demand: **{_fmt_mw(h.get('maximum_demand_mw'))}** at {h.get('maximum_date')} {h.get('maximum_hour'):02d}:00",
            f"Lowest demand: **{_fmt_mw(h.get('minimum_demand_mw'))}** at {h.get('minimum_date')} {h.get('minimum_hour'):02d}:00",
        ])

    def _weather(self, evidence):
        w = evidence.get("weather_summary", {})
        if not w.get("available"):
            return (
                "## 🌦 Weather Intelligence\n\n"
                f"{w.get('reason', 'Local weather data is unavailable.')}\n\n"
                "RTEDD will not invent weather causes without local weather evidence."
            )

        items = "\n".join(f"- **{k}:** {v}" for k, v in w.get("weather", {}).items())
        return (
            "## 🌦 Weather Intelligence\n\n"
            "### Latest locally available weather evidence\n"
            f"{items}\n\n"
            "Weather observations can be compared with demand, but correlation alone should not be presented as proven causation."
        )

    def _demand_change(self, evidence, findings):
        d = evidence.get("demand_change", {})
        if not d.get("available"):
            return f"## ⚡ Demand Investigation\n\n{d.get('reason', 'Demand evidence unavailable.')}"

        lines = [
            "## ⚡ Demand Investigation",
            "",
            f"Current demand: **{_fmt_mw(d.get('current_demand_mw'))}**",
            f"Previous hour: **{_fmt_mw(d.get('previous_demand_mw'))}**",
            f"Hourly change: **{_fmt_mw(d.get('change_mw'))} ({_fmt_pct(d.get('change_percent'))})**",
        ]

        if _number(d.get("expected_demand_mw")) is not None:
            lines += [
                f"Expected baseline: **{_fmt_mw(d.get('expected_demand_mw'))}**",
                f"Actual vs baseline: **{_fmt_pct(d.get('baseline_gap_percent'))}**",
            ]

        if findings:
            lines += ["", "### Key findings"]
            lines += [f"- {x['statement']}" for x in findings[:3]]

        lines += [
            "",
            "### What this means",
            "These findings describe observed demand behavior. A specific external cause requires supporting evidence.",
        ]
        return "\n".join(lines)

    def _system(self, evidence, findings):
        s = evidence.get("system_snapshot", {})
        if not s.get("available"):
            return "## ⚡ RTEDD System Status\n\nNo current RTEDD data is available."

        lines = [
            "## ⚡ RTEDD Intelligence Summary",
            "",
            f"**Current demand:** {_fmt_mw(s.get('current_demand_mw'))}",
            f"**Expected demand:** {_fmt_mw(s.get('expected_demand_mw'))}",
            f"**Latest observation:** {s.get('date')} {s.get('hour'):02d}:00",
            f"**Timezone:** {s.get('timezone')}",
        ]

        if findings:
            lines += ["", "### What needs attention"]
            lines += [f"- **{x['title']}:** {x['statement']}" for x in findings[:4]]

        return "\n".join(lines)


# ---------------------------------------------------------------------
# Main native agent
# ---------------------------------------------------------------------

class RTEDDAgenticAnalyst:
    """
    Main entry point.

    Example:
        tools = RTEDDTools(df, forecast_fn=compute_ensemble_forecast)
        agent = RTEDDAgenticAnalyst(tools)
        result = agent.ask("Why is demand increasing?")
    """

    def __init__(self, tools: RTEDDTools, context: Optional[AgentContext] = None):
        self.tools = tools
        self.context = context or AgentContext()
        self.intent_engine = IntentEngine()
        self.planner = InvestigationPlanner()
        self.finding_engine = FindingEngine()
        self.explainer = ExplanationEngine()

    def _execute_tool(self, tool_name: str, question: str) -> dict[str, Any]:
        try:
            if tool_name == "recent_anomalies":
                return self.tools.recent_anomalies(limit=5)
            if tool_name == "historical_summary":
                days = self._extract_days(question) or 7
                return self.tools.historical_summary(days=days)
            method = getattr(self.tools, tool_name)
            return method()
        except Exception as exc:
            return {
                "available": False,
                "reason": f"Internal tool error in {tool_name}: {type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _extract_days(question: str) -> Optional[int]:
        q = question.lower()
        match = re.search(r"(\d+)\s*day", q)
        if match:
            return max(1, min(int(match.group(1)), 365))
        if "week" in q:
            return 7
        if "month" in q:
            return 30
        return None

    def ask(self, question: str) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            return {
                "answer": "Please ask a question about RTEDD demand, forecasts, anomalies, history, weather, or system status.",
                "intent": None,
                "plan": [],
                "evidence": {},
                "findings": [],
            }

        intent, subject, scores = self.intent_engine.detect(question, self.context)
        plan = self.planner.create_plan(intent, question)

        evidence: dict[str, dict[str, Any]] = {}
        executed_steps = []

        for tool_name in plan:
            evidence[tool_name] = self._execute_tool(tool_name, question)
            executed_steps.append(tool_name)

        findings = self.finding_engine.build(evidence)
        answer = self.explainer.respond(question, intent, evidence, findings)

        # Update context using the latest available system date/hour.
        snapshot = evidence.get("system_snapshot", {})
        date = None
        hour = None
        if snapshot.get("date"):
            date = pd.to_datetime(snapshot["date"], errors="coerce")
        if snapshot.get("hour") is not None:
            hour = snapshot["hour"]

        self.context.remember(question, intent, subject, date=date, hour=hour)

        return {
            "answer": answer,
            "intent": intent,
            "subject": subject,
            "plan": executed_steps,
            "evidence": evidence,
            "findings": findings,
            "intent_scores": scores,
            "context": {
                "last_intent": self.context.last_intent,
                "last_subject": self.context.last_subject,
                "last_date": str(self.context.last_date) if self.context.last_date is not None else None,
                "last_hour": self.context.last_hour,
            },
        }


def suggested_questions() -> list[str]:
    return [
        "What's happening in the electricity system right now?",
        "Why is demand changing?",
        "What is the next-hour forecast?",
        "Compare Prophet, LightGBM and Ensemble.",
        "Explain the latest anomaly.",
        "Is today behaving normally compared with the forecast benchmark?",
        "Which live-trained model is performing best today?",
        "Summarize the last 7 days.",
    ]
