from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import StructuredTool
from langchain_community.document_loaders import TextLoader


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
        google_api_key="google api key"  # Replace with your actual key
    )

    # Define the QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )
 
    # Define tool using StructuredTool
    def gita_lookup(query: str) -> str:
        return qa_chain.run(query)

    gita_tool = StructuredTool.from_function(
        func=gita_lookup,
        name="BhagavadGitaSearch",
        description="Search Bhagavad Gita to find Krishna's teachings. Input should be a question like 'What does Krishna say about detachment?'"
    )

    # Initialize Agent
    agent = initialize_agent(
        tools=[gita_tool],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Invoke the agent with the question
    response = agent.invoke({"input": question})

    return response["output"]

