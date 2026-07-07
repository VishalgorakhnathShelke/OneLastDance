import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch


# =========================================================
# STEP 0: LOAD API KEYS
# =========================================================

# This loads keys from your .env file.
# Your .env should contain:
# OPENAI_API_KEY=your_openai_key
# GROQ_API_KEY=your_groq_key
# TAVILY_API_KEY=your_tavily_key
load_dotenv()


# =========================================================
# STEP 1: CREATE TOOLS
# =========================================================

# TavilySearch is a web search tool.
# The writer can use this if the topic needs current information.
search_tool = TavilySearch(max_results=3)

# LangGraph ToolNode expects a list of tools.
tools = [search_tool]


# =========================================================
# STEP 2: CREATE LLMs
# =========================================================

# # Writer LLM writes the LinkedIn post.
# writer_llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.7
# )
writer_llm =  ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


# This allows the writer model to call tools.
# Without bind_tools(), the writer cannot use Tavily search.
writer_llm_with_tools = writer_llm.bind_tools(tools)


# Reviewer LLM reviews the LinkedIn post.
# Lower temperature because reviewer should be strict and stable.
reviewer_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)


# =========================================================
# STEP 3: DEFINE STATE
# =========================================================

class State(TypedDict):
    """
    State is the shared memory of the graph.

    It stores:
    1. topic            = LinkedIn post topic
    2. messages         = conversation history and tool messages
    3. draft            = current LinkedIn post draft
    4. review_feedback  = feedback from reviewer
    5. is_approved      = True or False
    6. attempt          = how many times writer has tried

    add_messages means:
    New messages are added to old messages instead of replacing them.
    """

    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# =========================================================
# STEP 4: WRITER SYSTEM PROMPT
# =========================================================

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim with short paragraphs, around "
    "150 to 200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)


# =========================================================
# STEP 5: WRITER NODE
# =========================================================

def writer_node(state: State) -> dict:
    """
    Writer node writes or rewrites the LinkedIn post.

    It can do two things:
    1. Directly write the post
    2. Ask to use Tavily search first

    It reads:
    state["topic"]
    state["review_feedback"]
    state["attempt"]

    It returns:
    updated messages
    updated attempt count
    """

    print("\n--- Running Writer Node ---")

    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    # First attempt: write from topic.
    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic: {topic}. "
            f"If you need current information, search the web first."
        )

    # Later attempts: rewrite using reviewer feedback.
    else:
        user_message = (
            f"Your previous draft on '{topic}' was rejected.\n\n"
            f"Reviewer feedback:\n{previous_feedback}\n\n"
            f"Write a new improved draft that fixes every issue mentioned. "
            f"Do not repeat the same mistakes."
        )

    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", user_message)
    ]

    # This may return:
    # 1. a normal AIMessage with content
    # 2. an AIMessage containing tool_calls
    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages": [("human", user_message), response],
        "attempt": attempt
    }


# =========================================================
# STEP 6: TOOL NODE
# =========================================================

# ToolNode executes the tool call requested by writer_node.
# Example:
# writer asks TavilySearch to search latest AI trends.
# ToolNode runs the search and adds result to messages.
tool_node = ToolNode(tools)


# =========================================================
# STEP 7: EXTRACT DRAFT NODE
# =========================================================

def extract_draft_node(state: State) -> dict:
    """
    This node extracts the final LinkedIn post from the last AI message.

    It reads:
    state["messages"][-1].content

    It returns:
    {"draft": draft}
    """

    print("\n--- Running Extract Draft Node ---")

    last_message = state["messages"][-1]
    draft = last_message.content

    print(f"\nGenerated post:\n{draft}\n")

    return {
        "draft": draft
    }


# =========================================================
# STEP 8: REVIEWER SYSTEM PROMPT
# =========================================================

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim with short paragraphs\n"
    "4. Roughly 150 to 200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone, not corporate robotic\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)


# =========================================================
# STEP 9: REVIEWER NODE
# =========================================================

def reviewer_node(state: State) -> dict:
    """
    Reviewer node checks whether the draft is publish-ready.

    It reads:
    state["draft"]

    It returns:
    review_feedback
    is_approved
    """

    print("\n--- Running Reviewer Node ---")

    draft = state["draft"]

    prompt = (
        f"Review this LinkedIn post draft:\n\n"
        f"{draft}\n\n"
        f"Give your review."
    )

    response = reviewer_llm.invoke([
        ("system", REVIEWER_SYSTEM_PROMPT),
        ("human", prompt)
    ])

    review_text = response.content.strip()

    # Check only the verdict section.
    verdict_part = review_text.upper().split("FEEDBACK")[0]
    is_approved = "APPROVED" in verdict_part

    # Extract feedback.
    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"

    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved
    }


# =========================================================
# STEP 10: ROUTER 1
# SHOULD WRITER USE TOOL OR EXTRACT DRAFT?
# =========================================================

def should_use_tool(state: State):
    """
    This router checks the writer's last message.

    If writer requested a tool call:
    go to tools

    If writer produced final text:
    go to extract_draft
    """

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "extract_draft"


# =========================================================
# STEP 11: ROUTER 2
# SHOULD LOOP STOP OR REWRITE?
# =========================================================

def should_stop_looping(state: State):
    """
    This router decides what happens after review.

    If approved:
    stop

    If attempts reached 3:
    stop

    Otherwise:
    go back to writer for rewrite
    """

    if state["is_approved"]:
        print("\nPost has been approved.")
        return END

    if state["attempt"] >= 3:
        print("\nReached maximum attempts.")
        return END

    return "writer"


# =========================================================
# STEP 12: BUILD GRAPH
# =========================================================

graph = StateGraph(State)

# Add nodes.
graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node)

# Start at writer.
graph.add_edge(START, "writer")

# After writer, decide:
# If tool call exists, go to tools.
# Else, extract the draft.
graph.add_conditional_edges(
    "writer",
    should_use_tool,
    {
        "tools": "tools",
        "extract_draft": "extract_draft"
    }
)

# IMPORTANT:
# After tools run, go back to writer.
# Writer must read the tool result and generate the actual post.
graph.add_edge("tools", "writer")

# Once draft is extracted, review it.
graph.add_edge("extract_draft", "reviewer")

# After reviewer, decide:
# approved or max attempts -> END
# rejected and attempts left -> writer
graph.add_conditional_edges(
    "reviewer",
    should_stop_looping,
    {
        "writer": "writer",
        END: END
    }
)

# Compile graph.
app = graph.compile()


# =========================================================
# STEP 13: TERMINAL INTERFACE
# =========================================================

print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)

print("\nThis tool will:")
print("1. Draft a LinkedIn post")
print("2. Search the web if needed")
print("3. Review the draft")
print("4. Rewrite it if rejected")
print("5. Stop when approved or after 3 attempts")

print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")

else:
    print("\nStarting generation...\n")

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempt']}")
    print(f"Approved: {final_state['is_approved']}")
