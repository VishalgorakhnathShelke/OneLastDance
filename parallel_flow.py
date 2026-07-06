import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
# from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# This loads environment variables from the .env file.
# Example: GROQ_API_KEY will be loaded from .env.
load_dotenv()


# This creates the LLM object.
# We are using ChatGoogleGenerativeAI's hosted gemini model, so the model runs in gemini cloud.
# temperature=0.1 means the output will be more stable and less random.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.1)

def merge_score_dicts(existing: dict, newupdate: dict) -> dict:
    """
    This function is used to merge score dictionaries.

    Why do we need this?
    In this graph, multiple nodes run in parallel.
    All of them update the same state key: safety_scores.

    Example:
    toxicity_node returns:
    {"safety_scores": {"toxicity_level": 80}}

    copyright_node returns:
    {"safety_scores": {"copyright_risk": 60}}

    culture_node returns:
    {"safety_scores": {"cultural_insensitivity": 40}}

    Instead of replacing one score with another,
    this function combines all scores into one dictionary.
    """

    # If there is no old dictionary yet, just return the new one.
    if existing is None:
        return newupdate

    # Merge the old dictionary and new dictionary.
    # **existing unpacks the old values.
    # **newupdate unpacks the new values.
    # If the same key exists in both, newupdate will override existing.
    return {**existing, **newupdate}


# This defines the structure of the state.
# State is the shared memory of the graph.
# Every node can read from state and return updates to state.
class AnalyzerState(TypedDict):
    # raw_text stores the input text that we want to analyze.
    raw_text: str

    # safety_scores stores all safety-related scores.
    # Annotated tells LangGraph:
    # "When multiple nodes update safety_scores, use merge_score_dicts to combine them."
    safety_scores: Annotated[dict[str, int], merge_score_dicts]


def toxicity_node(state: AnalyzerState) -> dict:
    """
    This node checks the text for toxic content.

    It looks for:
    - profanity
    - aggression
    - hate speech
    - toxic language

    It returns one score:
    toxicity_level
    """

    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")

    # This prompt asks the LLM to give only one integer score from 0 to 100.
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    # Send the prompt to the LLM.
    response = llm.invoke(prompt)

    # Try to convert the LLM response into an integer.
    # Example: "75" becomes 75.
    # If conversion fails, set score to 0 so the program does not crash.
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return the result into the safety_scores dictionary.
    # Because safety_scores uses merge_score_dicts,
    # this score will be merged with other scores.
    return {
        "safety_scores": {
            "toxicity_level": score
        }
    }


def copyright_node(state: AnalyzerState) -> dict:
    """
    This node checks the text for copyright and originality risks.

    It looks for:
    - copied content
    - unoriginal content
    - corporate trademark risks

    It returns one score:
    copyright_risk
    """

    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")

    # This prompt asks the LLM to judge copyright/originality risk.
    # The model should return only a number from 0 to 100.
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    # Send the prompt to the LLM.
    response = llm.invoke(prompt)

    # Convert the response into an integer.
    # If the LLM gives something unexpected, use 0 as fallback.
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return copyright score inside the same safety_scores key.
    # This will be merged with toxicity and cultural scores.
    return {
        "safety_scores": {
            "copyright_risk": score
        }
    }


def culture_node(state: AnalyzerState) -> dict:
    """
    This node checks the text for cultural sensitivity risks.

    It looks for:
    - regional sensitivity issues
    - political risk
    - cultural insensitivity
    - content that may offend a global audience

    It returns one score:
    cultural_insensitivity
    """

    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")

    # This prompt asks the LLM to judge cultural sensitivity risk.
    # Again, it should only return a number from 0 to 100.
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    # Send the prompt to the LLM.
    response = llm.invoke(prompt)

    # Convert the LLM response into an integer.
    # If conversion fails, fallback to 0.
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return cultural sensitivity score inside safety_scores.
    # This gets merged with other safety scores.
    return {
        "safety_scores": {
            "cultural_insensitivity": score
        }
    }


# Create the LangGraph builder.
# AnalyzerState tells the graph what kind of state it will carry.
builder = StateGraph(AnalyzerState)


# Add all analysis nodes to the graph.
# First argument is the node name.
# Second argument is the Python function that should run.
builder.add_node("toxicity_node", toxicity_node)
builder.add_node("copyright_check", copyright_node)
builder.add_node("culture_node", culture_node)


# These edges start all 3 nodes from START.
# This means all three branches can run in parallel.
builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_check")
builder.add_edge(START, "culture_node")


# Each node goes directly to END after finishing.
# So the graph ends after all branches finish and their results are merged.
builder.add_edge("toxicity_node", END)
builder.add_edge("copyright_check", END)
builder.add_edge("culture_node", END)


# Compile the graph.
# After compiling, we get an app that can be executed using app.invoke().
app = builder.compile()


# This is the sample input text we want to analyze.
sample_script = """
Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
your friend's system using a script I copied directly from an online forum. 
Honestly, traditional security protocols are absolute garbage and anyone still using 
them is an absolute idiot. Let's dive into the code!
"""


# This is the initial state.
# raw_text contains the text to analyze.
# safety_scores starts as an empty dictionary.
# The three nodes will add their scores into this dictionary.
initial_state = {
    "raw_text": sample_script,
    "safety_scores": {}
}


# Run the graph.
# LangGraph sends the initial_state to all connected nodes.
# Each node returns its own score.
# merge_score_dicts combines all scores into safety_scores.
final_state = app.invoke(initial_state)


# Print the final merged safety scores.
print(final_state["safety_scores"])
