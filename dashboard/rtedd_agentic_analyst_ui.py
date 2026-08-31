
"""
Streamlit integration for RTEDD Agentic Analyst.

Put this file beside dashboard.py or copy render_rtedd_agentic_analyst()
into dashboard.py.
"""

import streamlit as st

from rtedd_agentic_analyst import (
    RTEDDTools,
    RTEDDAgenticAnalyst,
    suggested_questions,
)


def render_rtedd_agentic_analyst(df, forecast_fn=None, weather_df=None):
    st.markdown("## ⚡ RTEDD Agentic Analyst")
    st.caption(
        "A native RTEDD investigation engine. "
        "No OpenAI, API key, or external AI service is required."
    )

    # Build the agent using the current dashboard data.
    tools = RTEDDTools(
        df=df,
        forecast_fn=forecast_fn,
        weather_df=weather_df,
    )

    st.session_state.rtedd_agent = RTEDDAgenticAnalyst(tools)

    if "rtedd_agent_chat" not in st.session_state:
        st.session_state.rtedd_agent_chat = []

    # Chat history
    for message in st.session_state.rtedd_agent_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Suggested questions
    if not st.session_state.rtedd_agent_chat:
        st.markdown("### Investigate RTEDD")
        questions = suggested_questions()
        cols = st.columns(2)

        for i, question in enumerate(questions):
            if cols[i % 2].button(
                question,
                key=f"native_agent_suggestion_{i}",
                use_container_width=True,
            ):
                st.session_state["rtedd_agent_pending"] = question

    prompt = st.chat_input(
        "Ask about demand, forecasts, anomalies, benchmark behavior, or history..."
    )
    question = prompt or st.session_state.pop("rtedd_agent_pending", None)

    if question:
        st.session_state.rtedd_agent_chat.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Building investigation plan..."):
                result = st.session_state.rtedd_agent.ask(question)

            st.markdown(result["answer"])

            with st.expander("🧠 Investigation plan"):
                st.write(" → ".join(result["plan"]))
                st.caption(f"Detected intent: {result['intent']}")

            if result.get("findings"):
                with st.expander("📌 Ranked findings"):
                    for finding in result["findings"]:
                        st.markdown(
                            f"**{finding['title']}** ({finding['severity']})  \n"
                            f"{finding['statement']}"
                        )

            with st.expander("🔍 Evidence"):
                st.json(result["evidence"])

        st.session_state.rtedd_agent_chat.append(
            {"role": "assistant", "content": result["answer"]}
        )


# Example:
#
# elif view == "🤖 Agentic Analyst":
#     render_rtedd_agentic_analyst(
#         df=df_view,
#         forecast_fn=YOUR_EXISTING_FORECAST_FUNCTION,
#         weather_df=weather_df,
#     )
