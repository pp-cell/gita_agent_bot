from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import StructuredTool
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
import os
from langchain.agents import create_react_agent,AgentExecutor
from langchain.memory import ConversationBufferMemory

def indexing():
   
    print("Starting indexing process...")

    # Load and split the Bhagavad Gita text
    loader = TextLoader('geeta_text.txt', encoding='utf-8')
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Create embeddings and store in Chroma DB
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory="./gita_dbs"
    )

    print("Indexing complete. Chroma DB saved to './gita_dbs'")


# This function contains the logic for running the chatbot agent.
def gitabot(question: str):
    """
    Initializes a chatbot agent that uses the pre-existing Chroma DB
    to answer questions about the Bhagavad Gita.

    Args:
        question (str): The user's question.
    """
    print(f"\nInitializing Gita Chatbot for question: '{question}'")
    
    # Load the embedding model (same one used for indexing)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the pre-existing vector store from the persistent directory
    vectorestore = Chroma(
        embedding_function=embedding,  # Correct parameter is `embedding_function`
        persist_directory="./gita_dbs"
    )

    retriever = vectorestore.as_retriever()

    # Set up Gemini LLM
    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key="Api Key"  # Replace with your actual key
    )

    # Define the QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )
 
    # Define tool using StructuredTool
    def gita_lookup(query: str) -> str: 
        return qa_chain.invoke(query)

    gita_tool = StructuredTool.from_function(
        func=gita_lookup,
        name="BhagavadGitaSearch",
        description="Search Bhagavad Gita to find Krishna's teachings. Input should be a question like 'What does Krishna say about detachment?'"
    )
    
    
    os.environ["SERPAPI_API_KEY"] = "api key"


    search_api = SerpAPIWrapper()

    search_tool = Tool(
        name="google_search",
        func=search_api.run,
        description="Useful for answering questions about current events or the state of the world"
    )


    system_message = (
    "You are an AI assistant with expertise in the Bhagavad Gita and general knowledge. "
    "You have access to a 'gita_tool' for specific questions on the Bhagavad Gita and a 'search_tool' for all other questions. "
    "Your primary goal is to use the most relevant tool to answer the user's question accurately. "
    "Do not use the 'gita_tool' for general knowledge questions, and do not use the 'search_tool' for questions about the Gita. "
    "Always cite the relevant verse when using the 'gita_tool' if available."
     )

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "You have access to the following tools:\n{tools}\n\n"
            "Use the following format in your reasoning:\n"
            "Question: the input question you must answer\n"
            "Thought: your reasoning about what to do next\n"
            "Action: the action to take, must be one of [{tool_names}]\n"
            "Action Input: the input to that action\n"
            "Observation: the result of that action\n"
            "... (repeat Thought/Action/Observation as many times as needed)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the best answer to the original question\n\n"
            "Begin!\n\n"
            "Question: {input}\n"
            "{agent_scratchpad}"
        ),
    ]
 )

    # Initialize Agent
    agent=create_react_agent(
    llm=llm,
    tools=[gita_tool,search_tool],
    prompt=prompt
     )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    executor=AgentExecutor(
    agent=agent,
    tools=[gita_tool,search_tool],
    memory=memory,
    verbose=True
     )
    
    # Invoke the agent with the question
    response = executor.invoke({"input": question})

    return response["output"]

