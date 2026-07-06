import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages 
from langgraph.graph import StateGraph , START , END 
# from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv 

load_dotenv()


#Step 1 - Building the RAG retrievers 

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2" )



def build_retriever(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, 
                                              chunk_overlap = 100)
    
    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks,embeddings)

    return vectorstore.as_retriever(search_kwargs = {"k":4})

design_retriever = build_retriever("project-design.pdf")
final_project_retriever = build_retriever("project.pdf")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)


#step2 - State 

class State(TypedDict):
    project_topic : str  
    messages : Annotated[list,add_messages]   ### because we want to save the whole history
    query_type : str 
    retrieved_context : str  ### the answer from the RAG retriever

#Step 3 - Nodes generation 
    
def classifier_node(state: State) -> dict:
    """Classifies the student query into design, project, or general."""
    last_message = state['messages'][-1].content    
# . content gives us the actual text of the last message. insted of metadata like who sent it or when it was sent.
    prompt = (
        "Classify the following student query into exactly one category: "
        "'design', 'project', or 'general'.\n\n"
        "Use 'design' for questions regarding the project pitch, initial scope, mockups, "
        "user interfaces, initial dataset collection (Kaggle/HuggingFace), or Part 1 milestones.\n"
        "Use 'project' for questions about final implementation, Jupyter notebooks, team presentation guidelines, "
        "individual final report layout, evaluation metrics, plagiarism rules, deadlines, or grading rubrics.\n"
        "Use 'general' for greetings, casual talk, or queries completely unrelated to the COMP9727 project tasks.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: design, project, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "design" in category:
        category = "design"
    elif "project" in category:
        category = "project"
    else:
        category = "general"
    
    return {"query_type": category}

# create category nodes for each type of query

def design_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the Project Pitch & Design PDF."""
    query = state["messages"][-1].content
    docs = design_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def project_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the Project Implementation & Evaluation PDF."""
    query = state["messages"][-1].content
    docs = final_project_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def general_node(state: State) -> dict:
    """Answers directly using LLM internal knowledge."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def response_node(state: State) -> dict:
    """Generates the final structured response using course context and student topic."""
    query = state["messages"][-1].content
    project_topic = state.get("project_topic", "General Recommender System")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a helpful university tutor for COMP9727 (Recommender Systems). "
            f"The student is building a '{project_topic}' recommender. "
            f"Respond to their casual chat or query normally:\n\n{query}"
        )
    else:
        prompt = (
            f"You are an expert university course assistant for COMP9727 (Recommender Systems). "
            f"The student's chosen project domain is: '{project_topic}'.\n\n"
            f"Use the following official course criteria to answer their question precisely. "
            f"Apply the course rules (deadlines, submission constraints, formatting) directly to their context if applicable.\n"
            f"Do not guess figures, deadlines, or grade weights; stick strictly to the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Provide a clear, authoritative, yet friendly and actionable answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}
#step 4 - router function 
# ==========================================
# Step 4 - Router Function 
# ==========================================

def route_query(state: State):
    if state['query_type'] == 'design':
        return "design_rag"
    elif state['query_type'] == "project":
        return "project_rag"
    else:
        return "general"

# ==========================================
# Step 5 - Building the Graph 
# ==========================================

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("design_rag", design_rag_node)
graph.add_node("project_rag", project_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

# Edges 
graph.add_edge(START, "classifier")
graph.add_conditional_edges("classifier", route_query)
graph.add_edge("design_rag", "response")
graph.add_edge("project_rag", "response")
graph.add_edge("general", "response")
graph.add_edge("response", END)

app = graph.compile()

# ==========================================
# Step 6 - Terminal Interface
# ==========================================

print("=== Welcome to the COMP9727 Recommender Systems Project Assistant ===\n")
user_topic = input("What is your team's recommender project topic/domain? (e.g., Movie, Book, E-commerce): ")
if not user_topic.strip():
    user_topic = "General Recommender"

print(f"\nAwesome! System configured for a '{user_topic}' project context. Ask me anything about deadlines, report structures, or rules.\n")

while True:
    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit"]:
        print("Good luck with your COMP9727 project!")
        break
    
    result = app.invoke({
        "project_topic": user_topic,
        "messages": [("human", user_query)]
    })

    print(f"\nAssistant: {result['messages'][-1].content}\n" + "-"*40)