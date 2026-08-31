RTEDD Agentic Analyst

Native architecture

This implementation runs inside RTEDD and uses no:

OpenAI

API key

cloud LLM

external chatbot API

It is a bounded agentic system:

Question
   ↓
Intent Engine
   ↓
Investigation Planner
   ↓
Approved RTEDD Tools
   ↓
Evidence Engine
   ↓
Finding Ranking
   ↓
Grounded Explanation

Files

rtedd_agentic_analyst.py — core native agent

rtedd_agentic_analyst_ui.py — Streamlit UI integration

Dependencies

pip install pandas numpy

No AI API package is required.

Installation

Place both Python files in the same folder as dashboard.py:

dashboard/
├── dashboard.py
├── rtedd_agentic_analyst.py
└── rtedd_agentic_analyst_ui.py

Connect your existing forecast function

The agent accepts a local RTEDD function:

tools = RTEDDTools(
    df=df,
    forecast_fn=your_existing_forecast_function
)

The forecast dataframe should ideally contain:

Hour

Prophet

LightGBM

Ensemble

Ensemble_P10

Ensemble_P90

Important RTEDD rule

These remain separate:

Today vs Forecast Benchmark

Question:

Is today's demand behaving relative to the established expected baseline?

Uses:

Ontario Demand

Expected Demand

Today vs Live-Trained Forecast

Question:

How well are the live Prophet, LightGBM and Ensemble forecasts tracking today's actual demand?

Uses:

Ontario Demand

Prophet

LightGBM

Ensemble

The native agent never intentionally merges these concepts.

Current capabilities

Demand investigation

Checks:

current demand

previous hour

hourly movement

expected baseline

baseline deviation

Forecast investigation

Checks:

Prophet

LightGBM

Ensemble

P10/P90

model agreement

Anomaly investigation

Checks:

anomaly events

actual demand

expected demand

deviation

anomaly score

Historical analysis

Calculates:

average

maximum

minimum

selected period summary

Weather analysis

Uses only a locally supplied weather dataframe.

It does not invent weather causes.

Agentic behavior

For:

Why is demand changing?

The planner runs:

demand_change
→ benchmark_comparison
→ current_models
→ recent_anomalies

For:

Explain the latest anomaly

The planner runs:

recent_anomalies
→ demand_change
→ benchmark_comparison
→ current_models

Next development stage

The strongest next improvement is to add:

More advanced local NLP/entity extraction

Date-specific investigation

Cross-day comparisons

Proactive anomaly monitoring

Automated daily intelligence reports

A local retrieval/index layer for RTEDD documentation

Optional local/on-premise LLM support later, if desired