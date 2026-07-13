from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# =========================================================
# STEP 1: CONFIGURE THE KNOWLEDGE SOURCES
# =========================================================

# These PDFs are the two official sources used by the RAG workflow.
# Keep the files in the same folder as this script, or update these paths.
DESIGN_PDF_PATH = "project-design.pdf"
PROJECT_PDF_PATH = "project.pdf"

# This small sentence-transformer model is fast and works well for local
# semantic search over course documents.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings control how each PDF is broken into searchable pieces.
# Larger chunks give the LLM more context; overlap keeps related sentences
# together when a topic crosses a chunk boundary.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEARCH_RESULTS = 4

# Gemini is used for both routing and final answer generation.
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.4
EXIT_COMMANDS = {"exit", "quit"}

QueryType = Literal["design", "project", "general"]


# =========================================================
# STEP 2: DEFINE THE GRAPH STATE
# =========================================================

class State(TypedDict):
    """
    State is the shared memory passed between LangGraph nodes.

    project_topic:
        The student's recommender-system topic, for example "Movie" or
        "E-commerce". The response node uses it to personalize answers.

    messages:
        The conversation history. add_messages tells LangGraph to append new
        messages instead of replacing the old list.

    query_type:
        The category selected by the classifier node. It decides which route
        the graph follows next.

    retrieved_context:
        Text retrieved from the matching PDF. For general questions, this is
        set to NO_RETRIEVAL_NEEDED so the response node can answer directly.
    """

    project_topic: str
    messages: Annotated[list, add_messages]
    query_type: QueryType
    retrieved_context: str


# =========================================================
# STEP 3: CREATE SETUP FUNCTIONS
# =========================================================

def load_environment() -> None:
    """Load API keys and other settings from the local .env file."""
    load_dotenv()


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model used to convert PDF chunks into vectors."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model used by classifier and response nodes."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )


def build_retriever(pdf_path: str, embeddings: HuggingFaceEmbeddings):
    """
    Load one PDF, split it into chunks, store chunks in FAISS, and return a retriever.

    A retriever is the search component in RAG. It receives the student's
    question and returns the most relevant PDF chunks for the LLM to read.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": SEARCH_RESULTS})


def validate_pdf_file(pdf_path: str) -> None:
    """Fail early with a readable message if a required source PDF is missing."""
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(
            f"Required PDF not found: {pdf_path}. "
            "Place the file in the repository root or update the PDF path "
            "constant at the top of conditional_flow.py."
        )


def build_course_retrievers(embeddings: HuggingFaceEmbeddings) -> tuple:
    """Build both PDF retrievers used by the conditional graph."""
    validate_pdf_file(DESIGN_PDF_PATH)
    validate_pdf_file(PROJECT_PDF_PATH)

    design_retriever = build_retriever(DESIGN_PDF_PATH, embeddings)
    project_retriever = build_retriever(PROJECT_PDF_PATH, embeddings)
    return design_retriever, project_retriever


# =========================================================
# STEP 4: CREATE HELPER FUNCTIONS
# =========================================================

def get_latest_user_message(state: State) -> str:
    """Read the newest student message from the graph state."""
    return state["messages"][-1].content


def normalize_query_type(raw_category: str) -> QueryType:
    """
    Convert the LLM classifier output into one safe route name.

    The prompt asks for exactly one word, but this cleanup protects the graph
    if the model responds with extra text such as "Category: design".
    """
    category = raw_category.strip().lower()

    if "design" in category:
        return "design"

    if "project" in category:
        return "project"

    return "general"


def documents_to_context(documents: list[Document]) -> str:
    """Join retrieved PDF chunks into one context block for the answer prompt."""
    return "\n\n".join(document.page_content for document in documents)


def build_general_prompt(query: str, project_topic: str) -> str:
    """Create the prompt used when no PDF retrieval is needed."""
    return (
        "You are a helpful university tutor for COMP9727 "
        "(Recommender Systems). "
        f"The student is building a '{project_topic}' recommender. "
        "Respond to their casual chat or general query normally.\n\n"
        f"Student query: {query}"
    )


def build_rag_prompt(query: str, project_topic: str, context: str) -> str:
    """Create the prompt used when the answer must be grounded in PDF context."""
    return (
        "You are an expert university course assistant for COMP9727 "
        "(Recommender Systems). "
        f"The student's chosen project domain is: '{project_topic}'.\n\n"
        "Use the following official course criteria to answer their question "
        "precisely. Apply course rules such as deadlines, submission "
        "constraints, formatting, and grading rubrics directly to their "
        "context if applicable. Do not guess figures, deadlines, or grade "
        "weights; stick strictly to the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Provide a clear, authoritative, friendly, and actionable answer."
    )


def is_exit_command(user_query: str) -> bool:
    """Return True when the student wants to stop the terminal chat."""
    return user_query.strip().lower() in EXIT_COMMANDS


def print_assistant_reply(reply: str) -> None:
    """Print the assistant answer with a divider between turns."""
    print(f"\nAssistant: {reply}\n" + "-" * 40)


# =========================================================
# STEP 5: CREATE LANGGRAPH NODE FUNCTIONS
# =========================================================

def create_classifier_node(llm: ChatGoogleGenerativeAI):
    """
    Build the classifier node.

    This function returns the actual LangGraph node so the node can reuse the
    configured LLM without depending on a global variable.
    """

    def classifier_node(state: State) -> dict:
        """Classify the student query as design, project, or general."""
        query = get_latest_user_message(state)
        prompt = (
            "Classify the following student query into exactly one category: "
            "'design', 'project', or 'general'.\n\n"
            "Use 'design' for questions about the project pitch, initial "
            "scope, mockups, user interfaces, initial dataset collection "
            "(Kaggle/HuggingFace), or Part 1 milestones.\n"
            "Use 'project' for questions about final implementation, "
            "Jupyter notebooks, team presentation guidelines, individual "
            "final report layout, evaluation metrics, plagiarism rules, "
            "deadlines, or grading rubrics.\n"
            "Use 'general' for greetings, casual talk, or queries completely "
            "unrelated to the COMP9727 project tasks.\n\n"
            f"Query: {query}\n\n"
            "Return only one word: design, project, or general."
        )

        response = llm.invoke(prompt)
        return {"query_type": normalize_query_type(response.content)}

    return classifier_node


def create_design_rag_node(design_retriever):
    """Build the node that retrieves context from the design-stage PDF."""

    def design_rag_node(state: State) -> dict:
        query = get_latest_user_message(state)
        documents = design_retriever.invoke(query)
        return {"retrieved_context": documents_to_context(documents)}

    return design_rag_node


def create_project_rag_node(project_retriever):
    """Build the node that retrieves context from the final-project PDF."""

    def project_rag_node(state: State) -> dict:
        query = get_latest_user_message(state)
        documents = project_retriever.invoke(query)
        return {"retrieved_context": documents_to_context(documents)}

    return project_rag_node


def general_node(_state: State) -> dict:
    """
    Handle general messages without searching the PDFs.

    The response node still creates the final answer. This node only marks that
    retrieval was intentionally skipped.
    """
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def create_response_node(llm: ChatGoogleGenerativeAI):
    """Build the final response node that writes the assistant answer."""

    def response_node(state: State) -> dict:
        query = get_latest_user_message(state)
        project_topic = state.get("project_topic", "General Recommender System")
        context = state["retrieved_context"]

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = build_general_prompt(query, project_topic)
        else:
            prompt = build_rag_prompt(query, project_topic, context)

        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    return response_node


# =========================================================
# STEP 6: CREATE THE ROUTER FUNCTION
# =========================================================

def route_query(state: State) -> str:
    """
    Choose the next node after classification.

    LangGraph calls this after the classifier node. The returned string must
    match one of the graph node names registered in build_graph().
    """
    if state["query_type"] == "design":
        return "design_rag"

    if state["query_type"] == "project":
        return "project_rag"

    return "general"


# =========================================================
# STEP 7: BUILD AND COMPILE THE GRAPH
# =========================================================

def build_graph(llm, design_retriever, project_retriever):
    """
    Build the complete conditional RAG workflow.

    Flow:
    START -> classifier -> route_query()
        design  -> design_rag  -> response -> END
        project -> project_rag -> response -> END
        general -> general     -> response -> END
    """
    graph = StateGraph(State)

    graph.add_node("classifier", create_classifier_node(llm))
    graph.add_node("design_rag", create_design_rag_node(design_retriever))
    graph.add_node("project_rag", create_project_rag_node(project_retriever))
    graph.add_node("general", general_node)
    graph.add_node("response", create_response_node(llm))

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("design_rag", "response")
    graph.add_edge("project_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)

    return graph.compile()


# =========================================================
# STEP 8: RUN THE TERMINAL CHAT INTERFACE
# =========================================================

def ask_for_project_topic() -> str:
    """Ask the student for their recommender domain and provide a fallback."""
    user_topic = input(
        "What is your team's recommender project topic/domain? "
        "(e.g., Movie, Book, E-commerce): "
    )

    if not user_topic.strip():
        return "General Recommender"

    return user_topic.strip()


def run_terminal_chat(app) -> None:
    """Start the loop that sends user questions through the compiled graph."""
    print("=== Welcome to the COMP9727 Recommender Systems Project Assistant ===\n")
    user_topic = ask_for_project_topic()

    print(
        f"\nAwesome! System configured for a '{user_topic}' project context. "
        "Ask me anything about deadlines, report structures, or rules.\n"
    )

    while True:
        user_query = input("You: ")

        if is_exit_command(user_query):
            print("Good luck with your COMP9727 project!")
            break

        result = app.invoke(
            {
                "project_topic": user_topic,
                "messages": [("human", user_query)],
            }
        )

        print_assistant_reply(result["messages"][-1].content)


def main() -> None:
    """Prepare all dependencies, compile the graph, and launch the CLI app."""
    load_environment()
    embeddings = create_embeddings()
    llm = create_llm()
    design_retriever, project_retriever = build_course_retrievers(embeddings)
    app = build_graph(llm, design_retriever, project_retriever)
    run_terminal_chat(app)


if __name__ == "__main__":
    main()
