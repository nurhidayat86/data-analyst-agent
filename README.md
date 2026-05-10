# Data Analyst Agent

An AI-powered data analysis tool. Upload a CSV or Excel file, ask questions in
natural language, and the agent writes Python code, runs it in a sandbox, and
returns charts, narrative findings, and exportable Markdown reports / PowerPoint
slides.

Built on Google ADK patterns + [LiteLLM](https://github.com/BerriAI/litellm) +
[Streamlit](https://streamlit.io/). Model-agnostic — works with Claude, Gemini,
GPT, DeepSeek, Kimi, and local OpenAI-compatible servers (llama.cpp, vLLM).

## Architecture at a glance

```
Streamlit UI (streamlit_app.py)
        │
        ▼
Orchestrator ──► Analyst Agent ──► Code Executor (sandboxed exec)
                     │                    │
                     │                    ├─ pandas / sklearn / scipy / statsmodels
                     │                    └─ Plotly chart_generator
                     ▼
              Tool Registry (semantic search over examples/*.py)
                     │
                     ▼
            Report Writer (.md)  +  Slide Writer (.pptx)
```

Key modules:

- `adk/models/router.py` — LiteLLM wrapper, reads `config/models.yaml`.
- `adk/agents/orchestrator.py` — top-level agent, manages session state.
- `adk/agents/analyst.py` — generates and runs analysis code (3 retries on failure).
- `adk/tools/code_executor.py` — AST-checked import whitelist, restricted
  builtins, 60s threading timeout.
- `adk/tools/chart_generator.py` — Plotly chart factory (saves HTML to `output/charts/`).
- `adk/tools/schema_discovery.py` — auto-profiles uploaded DataFrames.
- `adk/tools/tool_registry.py` — scans `examples/` and `~/.adk-tools/` for
  `.py` files with YAML frontmatter and uses them as few-shot context.

## How to run

### 1. Prerequisites

- Python 3.10 or newer
- `pip` (or `uv` / `pipx` if you prefer)
- An API key for at least one supported provider, OR a local LLM server.

### 2. Clone and install

```bash
git clone <your-fork-or-repo-url> data-analyst-agent
cd data-analyst-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure model credentials

Pick whichever provider you want to use. The router reads keys from
environment variables — set the ones you need:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Claude
export GEMINI_API_KEY="..."               # Gemini
export OPENAI_API_KEY="sk-..."            # GPT-4o / GPT-4.1
export DEEPSEEK_API_KEY="..."             # DeepSeek
export KIMI_API_KEY="..."                 # Moonshot Kimi
```

You can also paste the key directly into the sidebar at runtime — handy for
quick experiments.

For **local models** (llama.cpp, vLLM, Ollama in OpenAI-compat mode):

```bash
export LOCAL_API_URL="http://localhost:8080/v1"   # or 8000 for vLLM
export LOCAL_API_KEY="anything"                   # most local servers ignore this
```

The default model and the full list of available models are declared in
`config/models.yaml`. Add or rename entries there if you want different
models in the sidebar dropdown.

### 4. Launch the app

```bash
streamlit run streamlit_app.py
```

Streamlit will open the UI at `http://localhost:8501`.

### 5. Use it

1. In the sidebar, pick a model and (if needed) paste an API key or base URL.
2. Upload a CSV or Excel file. The schema preview will appear.
3. Ask questions in the chat box, e.g.:
   - "Cluster customers by payment behavior."
   - "Show the distribution of transaction amounts by region."
   - "Which features predict churn?"
4. Charts render inline. After an analysis finishes, ask for a
   **"report"** to download a Markdown file, or **"slides"** to download a
   `.pptx` deck. Outputs are also saved to `output/`.

### 6. Add your own analysis tools (optional)

Drop a `.py` file into `examples/` (or `~/.adk-tools/`) with YAML frontmatter:

```python
---
description: Short summary the agent will match against requests
use_case: When this analysis is appropriate
tags: [clustering, segmentation]
---

# your reference implementation here — assumes `df` is in scope
import pandas as pd
from adk.tools.chart_generator import bar
...
```

The Tool Registry auto-discovers it on startup and surfaces it as few-shot
context for matching requests. See `examples/clustering.py`,
`examples/churn_analysis.py`, and `examples/default_analysis.py` as
templates.

## Troubleshooting

- **`Unknown model 'X'`** — check the name against `config/models.yaml`.
- **Auth errors** — confirm the matching env var is set (see the table in
  `adk/models/router.py`: `PROVIDER_ENV_VARS`) or paste the key in the sidebar.
- **`Forbidden imports detected`** — the sandbox blocks anything outside the
  whitelist in `adk/tools/code_executor.py`. Add the module to `ALLOWED_MODULES`
  if it is genuinely safe and required.
- **Local model unreachable** — verify `LOCAL_API_URL` points at a running
  OpenAI-compatible server (e.g. `curl $LOCAL_API_URL/models`).
- **Empty / truncated results** — the analyst retries up to 3 times; expand
  the "Show generated code" panel in the UI to inspect what the LLM produced.

## Project layout

```
data-analyst-agent/
├── streamlit_app.py        # Streamlit UI entry point
├── requirements.txt
├── config/
│   └── models.yaml         # Model registry
├── adk/
│   ├── agents/             # Orchestrator + Analyst
│   ├── models/             # LiteLLM router
│   └── tools/              # Sandbox, charts, schema, registry, writers
├── examples/               # Reference analysis tools (also few-shot context)
├── data/                   # Cached schema.json
└── output/                 # Generated charts, reports, slides
```
