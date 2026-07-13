# OneLastDance

OneLastDance is a collection of LangGraph and LangChain examples for building
LLM-powered workflows. The repository focuses on graph-based orchestration,
including conditional routing, parallel analysis, iterative review loops, and
project-assistant style RAG over course PDFs.

## Repository Contents

| File | Purpose |
| --- | --- |
| `conditional_flow.py` | Conditional RAG assistant for COMP9727 project questions. It classifies a query, routes it to the correct PDF retriever, and generates a grounded answer. |
| `parallel_flow.py` | Parallel LangGraph workflow that evaluates text across multiple safety dimensions and merges the scores. |
| `iterative_tools.py` | Iterative writer-reviewer workflow that can use tools such as web search before final approval. |
| `project_sequence.py` | Sequential project workflow example. |
| `langgraph_workflows.md` | Notes explaining common LangGraph workflow patterns. |
| `langgraph_workflows.html` | HTML version of the LangGraph workflow notes. |
| `langgraph_workflows.png` | Visual workflow reference image. |
| `requirement.txt` | Python dependency list for the project. |

## Main Workflow: `conditional_flow.py`

`conditional_flow.py` is organized into clear steps:

1. Configure PDF paths, embedding model, chunk sizes, and LLM settings.
2. Define the shared LangGraph state.
3. Create setup functions for environment loading, embeddings, LLM, and retrievers.
4. Create helper functions for reading messages, normalizing classifier output, and building prompts.
5. Create node functions for classification, PDF retrieval, general routing, and final response generation.
6. Route each query to the correct branch.
7. Build and compile the LangGraph workflow.
8. Run the terminal chat interface.

The graph follows this flow:

```text
START
  -> classifier
  -> route_query()
      -> design_rag  -> response -> END
      -> project_rag -> response -> END
      -> general     -> response -> END
```

## Requirements

- Python 3.10 or newer
- A Google Gemini API key for `ChatGoogleGenerativeAI`
- Local course PDFs for the conditional RAG assistant:
  - `project-design.pdf`
  - `project.pdf`

The PDF files should be placed in the repository root, or the constants in
`conditional_flow.py` should be updated with the correct paths.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirement.txt
```

Create a `.env` file in the repository root. You can start from the template:

```powershell
Copy-Item .env.example .env
```

Then replace the placeholder values:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Only the keys used by the script you are running are required. For example,
`conditional_flow.py` needs the Google key, while `iterative_tools.py` may also
need Tavily and Groq keys.

## Example Questions For `conditional_flow.py`

Try questions like:

- What should be included in our project design proposal?
- What are the final report expectations?
- How should we describe evaluation metrics for a movie recommender?
- Are there any plagiarism or submission rules we should remember?
- Hello, can you help us plan our recommender project?

## Running The Scripts

Run the conditional RAG assistant:

```powershell
python conditional_flow.py
```

Run the parallel safety-analysis workflow:

```powershell
python parallel_flow.py
```

Run the iterative writer-reviewer workflow:

```powershell
python iterative_tools.py
```

## Troubleshooting

If `conditional_flow.py` reports that a PDF is missing, confirm that
`project-design.pdf` and `project.pdf` exist in the repository root. These files
are ignored by git because they may be course materials or large local files.

If Gemini authentication fails, check that `.env` contains `GOOGLE_API_KEY` and
that the virtual environment is activated before running the script.

If dependency installation fails, upgrade pip first:

```powershell
python -m pip install --upgrade pip
```

## Development Notes

- Keep API keys in `.env`; do not commit secrets.
- Add new graph nodes as small functions so the workflow remains easy to test.
- Use docstrings to explain why a node exists and comments to explain each major
  step in the graph.
- Run a syntax check before committing Python changes:

```powershell
python -m py_compile conditional_flow.py
```

## Remote Notes

- need to add this
- wf
