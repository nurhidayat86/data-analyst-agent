"""Streamlit GUI for the Data Analyst Agent.

Features:
- Sidebar: model selector, credential inputs (API key / base URL),
  data upload (CSV/Excel), session history
- Main area: chat interface with inline Plotly plots
- Report and slides download buttons after analysis
- Tool registry viewer
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from plotly import io as plotly_io

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from adk.models.router import ModelRouter, PROVIDER_ENV_VARS
from adk.tools.schema_discovery import discover_schema, save_schema, load_schema
from adk.tools.tool_registry import ToolRegistry
from adk.tools.chart_generator import OUTPUT_DIR as CHARTS_OUTPUT_DIR
from adk.tools.report_writer import write_report
from adk.tools.slide_writer import write_slides
from adk.agents.orchestrator import Orchestrator
from adk.agents.analyst import AnalystAgent


# --- Page config ---
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
)


# --- Session state ---
def init_session():
    """Initialize session state variables."""
    if "router" not in st.session_state:
        st.session_state.router = ModelRouter()
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator(st.session_state.router)
    if "analyst" not in st.session_state:
        st.session_state.analyst = AnalystAgent(
            st.session_state.router,
            ToolRegistry(),
        )
    if "df" not in st.session_state:
        st.session_state.df = None
    if "schema" not in st.session_state:
        st.session_state.schema = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "last_report_path" not in st.session_state:
        st.session_state.last_report_path = None
    if "last_slide_path" not in st.session_state:
        st.session_state.last_slide_path = None
    if "model" not in st.session_state:
        st.session_state.model = st.session_state.router.default_model
    if "sidebar_api_key" not in st.session_state:
        st.session_state.sidebar_api_key = ""
    if "sidebar_api_base" not in st.session_state:
        st.session_state.sidebar_api_base = ""


init_session()


# --- Analysis function (defined early so it's available on all reruns) ---
def _run_analysis(prompt: str) -> None:
    """Run data analysis based on the user's prompt.

    Args:
        prompt: User's analysis request.
    """
    df = st.session_state.df
    schema = st.session_state.schema

    with st.spinner("Analyzing..."):
        result = st.session_state.analyst.analyze(
            request=prompt,
            df=df,
            schema=schema,
            model=st.session_state.model,
            api_key=st.session_state.sidebar_api_key or None,
            api_base=st.session_state.sidebar_api_base or None,
        )

    if result["success"]:
        # Store results
        st.session_state.analysis_results = result["results"]

        # Display narrative
        summary = result["results"].get("summary", "Analysis complete.")
        st.markdown(summary)

        # Display findings
        findings = result["results"].get("findings", [])
        if findings:
            st.markdown("**Key Findings:**")
            for i, finding in enumerate(findings, 1):
                st.markdown(f"{i}. {finding}")

        # Display charts
        charts = result["results"].get("charts", [])
        if charts:
            for chart_path in charts:
                chart_file = Path(chart_path)
                if chart_file.exists():
                    try:
                        fig = plotly_io.read_html(str(chart_file))[0]
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"chart-{chart_file.stem}",
                        )
                    except Exception:
                        st.caption(f"Chart: {chart_file.name}")

        # Store for report/slides
        st.session_state.last_report_path = None
        st.session_state.last_slide_path = None

        # Add to conversation history
        st.session_state.messages.append({
            "role": "assistant",
            "content": summary,
            "charts": charts,
        })
    else:
        error_msg = result.get("error", "Unknown error occurred")
        attempt = result.get("attempt", "?")
        code = result.get("code")
        st.error(f"Analysis failed after {attempt} attempt(s): {error_msg}")
        with st.expander("Show generated code"):
            if code:
                st.code(code, language="python")
            else:
                st.text("No code was generated.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Sorry, the analysis failed after {attempt} attempt(s): {error_msg}",
        })


# --- Sidebar ---
with st.sidebar:
    st.header("Data Analyst Agent")

    # Model selector
    st.subheader("Model", divider="blue")
    models = st.session_state.router.list_models()
    selected_model = st.selectbox(
        "LLM Model",
        options=models,
        index=models.index(st.session_state.model) if st.session_state.model in models else 0,
        help="Select the LLM model for analysis",
    )
    st.session_state.model = selected_model

    # Credentials
    st.subheader("Credentials", divider="blue")
    provider = st.session_state.router.get_provider(selected_model)
    if st.session_state.router.is_local_model(selected_model):
        st.text_input("API Key", type="password",
                       value=st.session_state.sidebar_api_key,
                       help="Optional for most local servers", key="sidebar_api_key")
        st.text_input("Base URL",
                       value=st.session_state.sidebar_api_base or st.session_state.router.get_model_spec(selected_model).get("base_url", ""),
                       key="sidebar_api_base")
    else:
        env_var_name = PROVIDER_ENV_VARS.get(provider, "UNKNOWN_API_KEY")
        st.text_input("API Key", type="password",
                       value=st.session_state.sidebar_api_key,
                       help=f"Set via {env_var_name} env var or enter here", key="sidebar_api_key")

    # Data upload
    st.subheader("Data", divider="blue")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Upload your dataset for analysis",
    )

    if uploaded_file is not None:
        # Read the uploaded file
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df
            st.session_state.schema = discover_schema(df)
            save_schema(st.session_state.schema)
            st.session_state.analysis_results = None
            st.session_state.last_report_path = None
            st.session_state.last_slide_path = None

            st.success(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
            with st.expander("View schema"):
                st.json(st.session_state.schema["shape"])
                st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {e}")
            uploaded_file = None

    # Session controls
    st.subheader("Session", divider="blue")
    if st.button("Clear Session", type="primary"):
        st.session_state.df = None
        st.session_state.schema = None
        st.session_state.analysis_results = None
        st.session_state.orchestrator.reset()
        st.rerun()

    # Tool registry
    st.subheader("Tools", divider="blue")
    tool_registry = ToolRegistry()
    all_tools = tool_registry.get_all()
    if all_tools:
        st.caption(f"{len(all_tools)} analysis tools available")
        with st.expander("Browse tools"):
            for tool in all_tools:
                st.markdown(f"**{tool['name']}** — {tool['description']}")
    else:
        st.caption("No analysis tools found. Add .py files to examples/ or ~/.adk-tools/")


# --- Main area ---
st.title("Data Analyst Agent")

if st.session_state.df is None:
    st.info("Upload a CSV or Excel file to get started. You can then ask questions about the data.")
else:
    df = st.session_state.df

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("charts"):
                for chart_path in message["charts"]:
                    chart_file = Path(chart_path)
                    if chart_file.exists():
                        st.plotly_chart(
                            plotly_io.read_html(str(chart_file))[0],
                            use_container_width=True,
                            key=str(chart_file),
                        )

    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the request
        with st.chat_message("assistant"):
            # Check if this is a report or slides request
            prompt_lower = prompt.lower()
            if "report" in prompt_lower or ".md" in prompt_lower:
                _handle_report_generation()
            elif "slide" in prompt_lower or ".pptx" in prompt_lower:
                _handle_slides_generation()
            else:
                # Run analysis
                result = _run_analysis(prompt)

        st.rerun()

    # Download buttons (if analysis completed)
    if st.session_state.last_report_path and Path(st.session_state.last_report_path).exists():
        with open(st.session_state.last_report_path, "rb") as f:
            st.download_button(
                label="Download Report (.md)",
                data=f,
                file_name=Path(st.session_state.last_report_path).name,
                mime="text/markdown",
            )

    if st.session_state.last_slide_path and Path(st.session_state.last_slide_path).exists():
        with open(st.session_state.last_slide_path, "rb") as f:
            st.download_button(
                label="Download Slides (.pptx)",
                data=f,
                file_name=Path(st.session_state.last_slide_path).name,
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )


def _handle_report_generation():
    """Generate a markdown report from the latest analysis results."""
    if st.session_state.analysis_results is None:
        st.warning("No analysis results available. Run an analysis first.")
        return

    results = st.session_state.analysis_results
    report_path = write_report(
        title="Data Analysis Report",
        summary=results.get("summary", "No summary available."),
        findings=results.get("findings", []),
        charts=results.get("charts", []),
        details=results.get("details", ""),
        model_used=st.session_state.model,
    )
    st.session_state.last_report_path = str(report_path)
    st.success(f"Report generated: {report_path.name}")
    st.markdown(f"Download: [{report_path.name}]({report_path})")

    # Show charts
    for chart_path in results.get("charts", []):
        chart_file = Path(chart_path)
        if chart_file.exists():
            st.plotly_chart(
                plotly_io.read_html(str(chart_file))[0],
                use_container_width=True,
                key=f"report-{chart_file.stem}",
            )


def _handle_slides_generation():
    """Generate PowerPoint slides from the latest analysis results."""
    if st.session_state.analysis_results is None:
        st.warning("No analysis results available. Run an analysis first.")
        return

    results = st.session_state.analysis_results
    slide_path = write_slides(
        title="Data Analysis Presentation",
        findings=results.get("findings", []),
        charts=results.get("charts", []),
        summary=results.get("summary", "No summary available."),
        model_used=st.session_state.model,
    )
    st.session_state.last_slide_path = str(slide_path)
    st.success(f"Slides generated: {slide_path.name}")
    st.markdown(f"Download: [{slide_path.name}]({slide_path})")


# --- Footer ---
st.caption("Data Analyst Agent — Powered by Google ADK + Streamlit")
