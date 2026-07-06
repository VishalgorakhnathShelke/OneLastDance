# import os 
# from typing import TypedDict

# #Lets create the state first 

# class state_pipeline(TypedDict):
#     raw_input : str 
#     edited_text : str 
#     script_text : str 
#     final_output : str 

# """state is something that is used to store the data and it is used to pass 
# the data from one node to another node
# and it is also used to store the final output of the graph and 
# it is also used to store the intermediate output of the graph and 
# it is also used to store the input of the graph and 
# it is also used to store the output of the graph and
#  it is also used to store the input"""

# # from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# load_dotenv()
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)

# #so basically state is some which keep 
# # update their self after running each nodes and it store the output in state dict
# # initially it will fetch the raw data from source and it will keep update after running each of the node


# # lets create the nodes 

# def editor_node(state :state_pipeline) -> dict:
#     """Stage 1: Cleans up grammar, removes typos, and refines the tone."""

#     prompt = (
#         "You are an expert copyeditor. Clean up the following raw text. "
#         "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow "
#         "while keeping the core message intact. Return only the edited text.\n\n"
#         f"Text:\n{state['raw_input']}"
#     )
#     response = llm.invoke(prompt)
#     #[ Your Prompt ] ---> .invoke() ---> [ Triggers API Call to OpenAI/Groq ]                                                
#     #[ Returns AIMessage Object ] <--------------------

#     return {"edited_text" : response.content.strip()}

# def scriptwriter_node(state: state_pipeline) -> dict:
#     """Stage 2: Formats the clean text into an engaging video script style."""
#     print("\n--- [Stage 2] Executing Scriptwriter Node ---")
    
#     prompt = (
#         "You are a charismatic YouTube content creator. Take this edited text and transform "
#         "it into a highly engaging, punchy, conversational video script hook. Make it sound "
#         "like a real person speaking passionately. Return only the script content.\n\n"
#         f"Edited Text:\n{state['edited_text']}"
#     )
    
#     response = llm.invoke(prompt)
#     return {"script_text": response.content.strip()}

# def translator_node(state: state_pipeline) -> dict:
#     """Stage 3: Translates the script into natural flowing Hinglish."""
#     print("\n--- [Stage 3] Executing Hinglish Translator Node ---")
    
#     prompt = (
#         "You are an expert content localizer for the Indian market. Take the following script "
#         "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
#         "or repeat information. Alternating comfortably between Hindi and English phrases just like "
#         "an intellectual tech educator would speak naturally on a live stream. Keep the energy high! "
#         "Return only the final Hinglish text.\n\n"
#         f"Script:\n{state['script_text']}"
#     )
    
#     response = llm.invoke(prompt)
#     return {"final_output": response.content.strip()}


# #now your state and nodes are ready and now it is time to create the graph 
# #and for creating the graph you have to connect tese nodes and for that you have 
# #to use the edges 
# #edges are very important to create the workflows 

# from langgraph.graph import StateGraph , START , END 




# #create the graph

# graph = StateGraph(state_pipeline)

# #add the nodes in our graph 
# # editor is name  for editor_node and scriptwriter is name for scriptwriter_node and translator is name for translator_node 

# graph.add_node("editor",editor_node)
# graph.add_node("scriptwriter",scriptwriter_node)
# graph.add_node("translator",translator_node)


# #Add edges (sequential - one after another)

# graph.add_edge(START,"editor")
# graph.add_edge('editor',"scriptwriter")
# graph.add_edge('scriptwriter',"translator")
# graph.add_edge('translator',END)

# #compile the graph 
# app = graph.compile()


# #giving the real messy data here 
# result = app.invoke({
#     "raw_input" :"AI agents are the future of tech. They can think, plan, and act on their own. LangGraph helps you build these agents with proper control and memory."
# })

# #output 
# print("your result are : - \n\n")
# print(result['final_output'])




# # so flow is like this
# # 1. raw_input is given to editor_node
# # 2. editor_node will clean the text and return the edited_text
# # 3. edited_text is given to scriptwriter_node
# # 4. scriptwriter_node will format the text into a video script and return the script_text
# # 5. script_text is given to translator_node
# # 6. translator_node will translate the script into Hinglish and return the final_output          



# # code flow is like we create a class for state here it is state_pipeline -> 
# # in which you describe the datatypes of each node: 
# # editor_node -> scriptwriter_node -> translator_node -> final_output

# # then we create nodes for each stage of the process 
# # and define the logic for each node
#     #[ Your Prompt ] ---> .invoke() ---> [ Triggers API Call to OpenAI/Groq ]                                                
#     #[ Returns AIMessage Object ] <--------------------

# # and node functions are given the state as input and return a 
# # dictionary with the output of that node  --- 
# # each time node will be the name of new key in dictionary and the value will be the output of that node

# # then we create a graph and the in graph function we called class state_pipeline 
# # graph = StateGraph(state_pipeline)
# # and then we add the nodes to that graph
# #graph.add_node("editor",editor_node)

# #  to the graph and then we add the edges to the graph and then we compile the graph and then we invoke the graph with the raw_input and then we get the final_output from the graph

# # as this is for sequencial code so we have used start and end 
# # to define the start and end of the graph and then we have added the edges
# #  to the graph to define the flow of the graph and then we have compiled 
# # the graph and then we have invoked the graph with the raw_input and then 
# # we have got the final_output from the graph

# #compile the graph 
# # app = graph.compile()

# #giving the real messy data here 
# # result = app.invoke({
# #     "raw_input" :"AI agents are the future of tech. They can think, plan, and act on their own. LangGraph helps you build these agents with proper control and memory."
# # })






from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# 1. LOAD API KEY
# ---------------------------------------------------------

# This loads your API key from the .env file.
# Your .env file should contain:
# GOOGLE_API_KEY=your_google_api_key_here
load_dotenv()


# ---------------------------------------------------------
# 2. CREATE THE LLM MODEL
# ---------------------------------------------------------

# This creates the Gemini model object.
# The model will run through Google API, not locally on your laptop.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)


# ---------------------------------------------------------
# 3. CREATE THE STATE
# ---------------------------------------------------------

class StatePipeline(TypedDict):
    """
    State is the shared memory of the graph.

    It stores:
    1. raw_input      = original user input
    2. edited_text    = output from editor_node
    3. script_text    = output from scriptwriter_node
    4. final_output   = output from translator_node

    Each node reads from this state and returns a dictionary.
    LangGraph then updates the state using that returned dictionary.
    """

    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


# ---------------------------------------------------------
# 4. CREATE NODE 1: EDITOR NODE
# ---------------------------------------------------------

def editor_node(state: StatePipeline) -> dict:
    """
    Stage 1: Editor Node

    This node reads:
    state["raw_input"]

    It does:
    Grammar correction, typo fixing, and tone improvement.

    It returns:
    {"edited_text": "..."}
    """

    print("\n--- [Stage 1] Executing Editor Node ---")

    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix grammatical errors, spelling mistakes, and improve the transition flow. "
        "Keep the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )

    # This sends the prompt to Gemini.
    # response will be an AIMessage object.
    response = llm.invoke(prompt)

    # We return only the updated part of the state.
    # LangGraph will merge this into the full state.
    return {
        "edited_text": response.content.strip()
    }


# ---------------------------------------------------------
# 5. CREATE NODE 2: SCRIPTWRITER NODE
# ---------------------------------------------------------

def scriptwriter_node(state: StatePipeline) -> dict:
    """
    Stage 2: Scriptwriter Node

    This node reads:
    state["edited_text"]

    It does:
    Converts the cleaned text into an engaging YouTube-style script.

    It returns:
    {"script_text": "..."}
    """

    print("\n--- [Stage 2] Executing Scriptwriter Node ---")

    prompt = (
        "You are a charismatic YouTube content creator. Take this edited text and transform "
        "it into a highly engaging, punchy, conversational video script hook. "
        "Make it sound like a real person speaking passionately. "
        "Return only the script content.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {
        "script_text": response.content.strip()
    }


# ---------------------------------------------------------
# 6. CREATE NODE 3: TRANSLATOR NODE
# ---------------------------------------------------------

def translator_node(state: StatePipeline) -> dict:
    """
    Stage 3: Translator Node

    This node reads:
    state["script_text"]

    It does:
    Converts the script into natural Hinglish.

    It returns:
    {"final_output": "..."}
    """

    print("\n--- [Stage 3] Executing Hinglish Translator Node ---")

    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing Hinglish. Do not translate sentence by sentence. "
        "Use a natural mix of Hindi and English, like an intellectual tech educator speaking "
        "on a live stream. Keep the energy high. Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {
        "final_output": response.content.strip()
    }


# ---------------------------------------------------------
# 7. CREATE THE GRAPH
# ---------------------------------------------------------

# StateGraph creates the workflow.
# StatePipeline tells LangGraph what kind of state this graph will carry.
graph = StateGraph(StatePipeline)


# ---------------------------------------------------------
# 8. ADD NODES TO THE GRAPH
# ---------------------------------------------------------

# "editor" is the graph node name.
# editor_node is the actual Python function.
graph.add_node("editor", editor_node)

# "scriptwriter" is the graph node name.
# scriptwriter_node is the actual Python function.
graph.add_node("scriptwriter", scriptwriter_node)

# "translator" is the graph node name.
# translator_node is the actual Python function.
graph.add_node("translator", translator_node)


# ---------------------------------------------------------
# 9. ADD EDGES TO DEFINE THE FLOW
# ---------------------------------------------------------

# Edges decide the order of execution.
# This is a sequential workflow.

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)


# ---------------------------------------------------------
# 10. COMPILE THE GRAPH
# ---------------------------------------------------------

# compile() converts the graph into a runnable application.
app = graph.compile()


# ---------------------------------------------------------
# 11. RUN THE GRAPH
# ---------------------------------------------------------

# This is the input text.
# We initialise all state keys to keep it beginner-friendly.
initial_state = {
    "raw_input": (
        "AI agents are the future of tech. They can think, plan, and act on their own. "
        "LangGraph helps you build these agents with proper control and memory."
    ),
    "edited_text": "",
    "script_text": "",
    "final_output": ""
}


# app.invoke() starts the graph.
# The graph follows this path:
# START -> editor -> scriptwriter -> translator -> END
result = app.invoke(initial_state)


# ---------------------------------------------------------
# 12. PRINT THE FINAL OUTPUT
# ---------------------------------------------------------

print("\nYour final result is:\n")
print(result["final_output"])


# ---------------------------------------------------------
# SIMPLE FLOW SUMMARY
# ---------------------------------------------------------

# 1. raw_input is given in the initial state.
# 2. editor_node reads raw_input and returns edited_text.
# 3. LangGraph updates the state with edited_text.
# 4. scriptwriter_node reads edited_text and returns script_text.
# 5. LangGraph updates the state with script_text.
# 6. translator_node reads script_text and returns final_output.
# 7. LangGraph updates the state with final_output.
# 8. We print result["final_output"].