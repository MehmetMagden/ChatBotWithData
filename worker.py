import os
import torch
import logging

# Configure application logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Prompt template ensuring the LLM uses retrieved context directly
QA_PROMPT_TEMPLATE = """You are a helpful AI assistant answering questions about the uploaded document.
Use the following pieces of context to answer the question at the end.
If you do not know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
)



# Device configuration for vector embedding generation
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global state variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None


def init_llm():
    """
    Initialize the LLM engine and text embeddings model.
    Uses Google Gemini for reasoning and Hugging Face for local vector representations.
    """
    global llm_hub, embeddings

    logger.info("Initializing Google Gemini LLM and local HuggingFace embeddings...")

    # Initialize Google Gemini using LangChain integration
    # Reads the GEMINI_API_KEY environment variable by default
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable is not set. Please ensure it is exported.")

    llm_hub = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2,
        max_tokens=1024,
        google_api_key=api_key
    )
    logger.debug("ChatGoogleGenerativeAI initialized: %s", llm_hub)

    # Initialize Hugging Face embeddings model for document chunk representation
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )
    logger.debug("Embeddings initialized with compute device: %s", DEVICE)


def process_document(document_path):
    """
    Process an uploaded PDF document:
    1. Parse and extract text with PyPDFLoader.
    2. Split content into contextual chunks.
    3. Index chunks into an in-memory Chroma vector store.
    4. Construct a RetrievalQA chain.
    """
    global conversation_retrieval_chain

    logger.info("Loading document from target path: %s", document_path)

    # Load PDF file
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    logger.debug("Loaded %d document page(s)", len(documents))

    # Split document into chunks with size of 1024 and overlap of 64
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    logger.debug("Document partitioned into %d chunk(s)", len(texts))

    # Create Chroma vector database using the pre-loaded embeddings
    logger.info("Generating embeddings and building Chroma vector index...")
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Chroma vector store successfully created.")

    # Build the RetrievalQA chain
    # MMR (Maximal Marginal Relevance) is used to balance relevance and diversity
# Build the QA chain with explicit prompt template
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question",
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    logger.info("RetrievalQA chain pipeline constructed successfully with custom prompt.")


def process_prompt(prompt):
    """
    Handle user query, retrieve relevant document segments,
    query the model with contextual chat history, and update message store.
    """
    global conversation_retrieval_chain
    global chat_history

    if conversation_retrieval_chain is None:
        return "Please upload and process a PDF document before asking questions."

    logger.info("Processing user prompt: %s", prompt)

    # Contextualize prompt with recent conversation history if available
    if chat_history:
        history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in chat_history[-3:])
        full_prompt = f"Previous conversation:\n{history_text}\n\nNew question: {prompt}"
    else:
        full_prompt = prompt

    # Invoke retrieval chain
    output = conversation_retrieval_chain.invoke({"question": full_prompt})
    answer = output.get("result", "No response generated.")
    logger.debug("Generated model answer: %s", answer)

    # Append prompt and answer tuple to conversation log
    chat_history.append((prompt, answer))
    logger.debug("Updated chat history. Total records: %d", len(chat_history))

    return answer


# Automatically initialize the LLM and embedding models on module load
init_llm()
logger.info("Worker initialization complete.")