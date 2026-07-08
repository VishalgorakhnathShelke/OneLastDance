# Onelastdance LangGraph Workflow Diagrams

## Overall LangGraph Pattern

```mermaid
flowchart TD
    A["1. Define State<br/>TypedDict"] --> B["2. Create Nodes<br/>Python functions"]
    B --> C["3. Build Graph<br/>StateGraph(State)"]
    C --> D["4. Add Nodes<br/>graph.add_node(...)"]
    D --> E["5. Connect Flow<br/>add_edge / add_conditional_edges"]
    E --> F["6. Compile<br/>app = graph.compile()"]
    F --> G["7. Run<br/>app.invoke(initial_state)"]
    G --> H["8. Final State<br/>result dictionary"]

    classDef setup fill:#e8f3ff,stroke:#1b6aa8,color:#0b2740
    classDef run fill:#eaf7ea,stroke:#2f7d32,color:#123b14
    class A,B,C,D,E,F setup
    class G,H run
```

## `project_sequence.py` - Simple Sequential Pipeline

```mermaid
flowchart LR
    START(["START"]) --> I["Initial State<br/>raw_input"]
    I --> E["editor_node<br/>Fix grammar and tone"]
    E --> ET["State update<br/>edited_text"]
    ET --> S["scriptwriter_node<br/>Make YouTube-style script"]
    S --> ST["State update<br/>script_text"]
    ST --> T["translator_node<br/>Convert to Hinglish"]
    T --> FO["State update<br/>final_output"]
    FO --> END(["END"])

    classDef state fill:#fff4d6,stroke:#b7791f,color:#3d2b00
    classDef node fill:#e8f3ff,stroke:#1b6aa8,color:#0b2740
    classDef terminal fill:#ececec,stroke:#555,color:#111
    class I,ET,ST,FO state
    class E,S,T node
    class START,END terminal
```

## `parallel_flow.py` - Parallel Safety Analyzer

```mermaid
flowchart TD
    START(["START"]) --> R["Initial State<br/>raw_text + safety_scores = {}"]

    R --> T["toxicity_node<br/>Toxicity score"]
    R --> C["copyright_check<br/>Copyright risk"]
    R --> U["culture_node<br/>Cultural sensitivity score"]

    T --> M["merge_score_dicts<br/>combine all branch outputs"]
    C --> M
    U --> M

    M --> F["Final State<br/>safety_scores = { toxicity_level,<br/>copyright_risk,<br/>cultural_insensitivity }"]
    F --> END(["END"])

    classDef state fill:#fff4d6,stroke:#b7791f,color:#3d2b00
    classDef node fill:#e8f3ff,stroke:#1b6aa8,color:#0b2740
    classDef merge fill:#f0e8ff,stroke:#6b46c1,color:#25124d
    classDef terminal fill:#ececec,stroke:#555,color:#111
    class R,F state
    class T,C,U node
    class M merge
    class START,END terminal
```

## `conditional_flow.py` - Conditional RAG Assistant

```mermaid
flowchart TD
    START(["START"]) --> Q["User Query<br/>messages[-1]"]
    Q --> CL["classifier_node<br/>Classify query"]

    CL --> D{"query_type?"}
    D -->|"design"| DR["design_rag_node<br/>Search project-design.pdf"]
    D -->|"project"| PR["project_rag_node<br/>Search project.pdf"]
    D -->|"general"| G["general_node<br/>No retrieval needed"]

    DR --> CTX["retrieved_context"]
    PR --> CTX
    G --> CTX

    CTX --> R["response_node<br/>Generate final answer"]
    R --> A["AI message added<br/>to messages"]
    A --> END(["END"])

    classDef state fill:#fff4d6,stroke:#b7791f,color:#3d2b00
    classDef node fill:#e8f3ff,stroke:#1b6aa8,color:#0b2740
    classDef decision fill:#ffe8e8,stroke:#c53030,color:#4a1111
    classDef terminal fill:#ececec,stroke:#555,color:#111
    class Q,CTX,A state
    class CL,DR,PR,G,R node
    class D decision
    class START,END terminal
```

## `iterative_tools.py` - Writer, Search Tool, Reviewer Loop

```mermaid
flowchart TD
    START(["START"]) --> INIT["Initial State<br/>topic, messages, draft,<br/>review_feedback, is_approved, attempt"]
    INIT --> W["writer_node<br/>Write or rewrite post<br/>attempt = attempt + 1"]

    W --> TC{"Did writer request<br/>TavilySearch tool?"}
    TC -->|"Yes"| TOOL["ToolNode<br/>Run Tavily search"]
    TOOL --> W

    TC -->|"No"| EX["extract_draft_node<br/>Save last AI message as draft"]
    EX --> REV["reviewer_node<br/>Approve or reject draft"]

    REV --> STOP{"Approved OR<br/>attempt >= 3?"}
    STOP -->|"No"| FB["Save review_feedback<br/>Loop back to writer"]
    FB --> W
    STOP -->|"Yes"| END(["END<br/>Print final post"])

    classDef state fill:#fff4d6,stroke:#b7791f,color:#3d2b00
    classDef node fill:#e8f3ff,stroke:#1b6aa8,color:#0b2740
    classDef tool fill:#eaf7ea,stroke:#2f7d32,color:#123b14
    classDef decision fill:#ffe8e8,stroke:#c53030,color:#4a1111
    classDef terminal fill:#ececec,stroke:#555,color:#111
    class INIT,FB state
    class W,EX,REV node
    class TOOL tool
    class TC,STOP decision
    class START,END terminal
```

