import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

logger = logging.getLogger(__name__)


class RAGAnalyzer:
    """
    A class to handle Retrieval-Augmented Generation (RAG) tasks.
    It indexes PDF documents from a folder and answers questions based on them.
    """

    def __init__(self, pdf_folder: str, persist_directory: str):
        """
        Initializes the RAG Analyzer.

        Args:
            pdf_folder (str): The folder where PDF files are stored.
            persist_directory (str): The directory to save/load the Chroma vectorstore.
        """
        self.pdf_folder = pdf_folder
        self.persist_directory = persist_directory

        # Initialize core LangChain components
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

        self.vectorstore = None
        self.qa_chain = None

        # Create directories if they don't exist
        os.makedirs(self.pdf_folder, exist_ok=True)
        os.makedirs(self.persist_directory, exist_ok=True)

    def _load_and_split_documents(self) -> list:
        """Loads all PDFs from the folder and splits them into chunks."""
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_folder}. The RAG tool will have no knowledge.")
            return []

        logger.info(f"Found {len(pdf_files)} PDF(s) to process.")

        all_docs = []
        for pdf_file in pdf_files:
            file_path = os.path.join(self.pdf_folder, pdf_file)
            try:
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load())
            except Exception as e:
                logger.error(f"Failed to load or process {pdf_file}: {e}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs_split = text_splitter.split_documents(all_docs)
        logger.info(f"Total text chunks from all PDFs: {len(docs_split)}")
        return docs_split

    def index_documents(self):
        """
        Creates and persists a new vectorstore from the documents in the PDF folder.
        This is a potentially long-running operation.
        """
        docs_split = self._load_and_split_documents()
        if not docs_split:
            logger.error("No documents were loaded or split. Cannot create index.")
            return

        logger.info("Building and persisting Chroma vectorstore... (this may take a while)")
        # Create the vectorstore from documents, which will be saved to disk
        self.vectorstore = Chroma.from_documents(
            documents=docs_split,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        logger.info("Vectorstore created and persisted successfully.")
        self._prepare_qa_chain()

    def load_or_create_vectorstore(self):
        """
        Loads the vectorstore from disk if it exists, otherwise creates it.
        This is called on server startup.
        """
        # Check if the vectorstore directory has files in it
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            logger.info(f"Loading existing vectorstore from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            logger.warning("No existing vectorstore found. Indexing new documents.")
            self.index_documents()

        if self.vectorstore:
            self._prepare_qa_chain()

    def _prepare_qa_chain(self):
        """Initializes the RetrievalQA chain."""
        if not self.vectorstore:
            logger.error("Cannot prepare QA chain: vectorstore is not initialized.")
            return

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" is a simple and effective method
            retriever=self.vectorstore.as_retriever()
        )
        logger.info("RAG QA chain is ready.")

    def ask(self, query: str) -> str:
        """Asks a question to the RAG chain."""
        if not self.qa_chain:
            return "Error: The Question-Answering system is not ready. Please check server logs."

        logger.info(f"RAG query received: '{query}'")
        try:
            result = self.qa_chain.invoke({"query": query})
            return result.get('result', "Sorry, I couldn't find an answer.")
        except Exception as e:
            logger.error(f"Error during RAG query processing: {e}", exc_info=True)
            return "An error occurred while trying to find an answer."