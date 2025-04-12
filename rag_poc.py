#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My GURU RAG System with AutoGen
Using Chain of Thought reasoning to improve answers

TODO: Add more detailed documentation
TODO: Improve chunking strategy
TODO: Add support for more document types
"""

import os
import sys
import json
import time
import logging
import requests
from typing import List, Dict, Any
import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
import autogen
from autogen import AssistantAgent, UserProxyAgent, Agent, ConversableAgent

# Project configuration
# TODO: Move this to a config file
PROJECT_NAME = "guru-ag-rag-poc-local"
PROJECT_ROOT = os.path.expanduser(f"~/repos/{PROJECT_NAME}")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
#smaller mode - didnt give good results on retreival
# TODO: explore embedding fine tuning
# EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/embeddings/bge-small-en")
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/embeddings//bge-large-en-v1.5")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "vector_db")
WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "workspace")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ollama configuration
# TODO: Test with other models like llama2 and mistral-instruct
OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_MODEL = "mistral:7b"  # Using locally downloaded model
OLLAMA_MODEL = "mistral:instruct"  # Using locally downloaded model
# OLLAMA_MODEL = "gemma:2b"  # Using locally downloaded model

# Ensure directories exist
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging - putting everything in file to keep the UI clean
# TODO: Add option for console logging during debug
log_filename = os.path.join(LOG_DIR, f"guru_rag_poc.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Starting my guru-autogen-rag-poc System")
logger.info(f"Log file: {log_filename}")


# Custom Ollama LLM class for CoT
# TODO: Refactor this to support multiple LLM backends
class OllamaLLM:
    def __init__(self, model_name=OLLAMA_MODEL):
        self.model_name = model_name
        self.api_url = f"{OLLAMA_BASE_URL}/api/generate"
        logger.info(f"Initializing OllamaLLM with model: {model_name}")

        # Test connection
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in available models: {model_names}")
                    print(
                        f"Warning: {self.model_name} not found in available models. Please run 'ollama pull {self.model_name}' first.")
                else:
                    logger.info(f"Successfully connected to Ollama. Using {self.model_name}.")
                    print(f"Successfully connected to Ollama model: {self.model_name}")
            else:
                logger.error(f"Error connecting to Ollama API: Status code {response.status_code}")
                print("Error connecting to Ollama. See log file for details.")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            print("Error connecting to Ollama. Make sure Ollama is running with 'ollama serve'")

    def __call__(self, messages, **kwargs):
        logger.info(f"OllamaLLM received call with {len(messages)} messages")

        # Log the input messages (truncated for readability)
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            truncated = content[:500] + "..." if len(content) > 500 else content
            logger.info(f"Message {i + 1} - Role: {msg.get('role', 'unknown')}, Content: {truncated}")

        # Convert chat messages to prompt
        prompt = self._messages_to_prompt(messages)
        logger.info(f"Generated prompt for Ollama (length: {len(prompt)})")
        logger.debug(f"Full prompt: {prompt}")

        # Call Ollama API
        try:
            logger.info(f"Calling Ollama API...")
            start_time = time.time()

            api_params = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.95),
                    "top_k": kwargs.get("top_k", 40),
                    "num_predict": kwargs.get("max_tokens", 1024)  # Increased for CoT
                }
            }
            logger.info(f"API parameters: {json.dumps({k: v for k, v in api_params.items() if k != 'prompt'})}")

            response = requests.post(
                self.api_url,
                json=api_params
            )

            elapsed_time = time.time() - start_time
            logger.info(f"Ollama API call completed in {elapsed_time:.2f} seconds")

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                truncated_response = response_text[:500] + "..." if len(response_text) > 500 else response_text
                logger.info(f"Received response from Ollama (length: {len(response_text)})")
                logger.info(f"Response preview: {truncated_response}")
                return {"content": response_text}
            else:
                logger.error(f"Error from Ollama API: {response.status_code}")
                logger.error(f"Error details: {response.text}")
                print("Error from Ollama API. See log file for details.")
                return {"content": "Error generating response from Ollama."}

        except Exception as e:
            logger.error(f"Exception calling Ollama API: {e}")
            print(f"Error: {str(e)}")
            return {"content": f"Error: {str(e)}"}

    def _messages_to_prompt(self, messages):
        """Convert messages to Mistral format with CoT instructions"""
        logger.info("Converting messages to prompt format with CoT instructions")
        prompt = ""

        # Add a system message at the beginning if not already there
        # TODO: Find a better way to inject CoT instructions
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            logger.info("Adding default system message with CoT instructions")
            cot_system_message = (
                "<s>You are an assistant that helps users find information in their documents. "
                "When answering questions, think step by step and reason through your thought process. "
                "Show your detailed chain of thought before giving a final answer. "
                "Label your thinking steps clearly with [THINKING] and your final answer with [ANSWER].</s>\n"
            )
            prompt += cot_system_message

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                # Add CoT instructions to the system message
                logger.info("Processing system message")
                if "chain of thought" not in content.lower() and "step by step" not in content.lower():
                    logger.info("Adding CoT instructions to system message")
                    content += "\nWhen answering questions, think step by step and reason through your thought process. Show your detailed chain of thought before giving a final answer. Label your thinking steps clearly with [THINKING] and your final answer with [ANSWER]."
                prompt += f"<s>{content}</s>\n"
            elif role == "user":
                # Add CoT instructions to the user prompt
                logger.info("Processing user message")
                if not content.startswith("Use the following") and "chain of thought" not in content.lower():
                    logger.info("Adding CoT instructions to user message")
                    prompt += f"<user>{content}\nShow your chain of thought reasoning step by step before providing your final answer.</user>\n"
                else:
                    prompt += f"<user>{content}</user>\n"
            elif role == "assistant":
                logger.info("Processing assistant message")
                prompt += f"<assistant>{content}</assistant>\n"

        prompt += "<assistant>"
        logger.info(f"Final prompt format created with length {len(prompt)}")
        return prompt


# Configure embedding model
# TODO: Try different embedding models
def setup_embedding_model():
    """Load or download the embedding model"""
    logger.info("Setting up embedding model")
    try:
        if os.path.exists(EMBEDDING_MODEL_PATH):
            logger.info(f"Loading embedding model from {EMBEDDING_MODEL_PATH}...")
            print(f"Loading embedding model...")
            return SentenceTransformer(EMBEDDING_MODEL_PATH)
        else:
            logger.info(f"Downloading embedding model to {EMBEDDING_MODEL_PATH}...")
            print(f"Downloading embedding model (this may take a while)...")
            os.makedirs(os.path.dirname(EMBEDDING_MODEL_PATH), exist_ok=True)
            model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            model.save(EMBEDDING_MODEL_PATH)
            return model
    except Exception as e:
        logger.error(f"Error setting up embedding model: {e}")
        print(f"Error setting up embedding model. See log for details.")
        sys.exit(1)


# Document loading function
# TODO: Add support for more document types (docx, txt, etc)
def load_documents(directory_path: str) -> List[Dict[str, Any]]:
    """Load PDF documents from a directory"""
    logger.info(f"Loading documents from {directory_path}")
    documents = []

    if not os.path.exists(directory_path):
        logger.warning(f"Directory not found: {directory_path}")
        print(f"Directory not found: {directory_path}")
        return documents

    found_files = False
    print("Loading documents...")
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            found_files = True
            file_path = os.path.join(directory_path, filename)
            logger.info(f"Processing PDF file: {file_path}")
            try:
                doc = pymupdf.open(file_path)
                logger.info(f"Successfully opened {filename} with {doc.page_count} pages")

                for page_num in range(doc.page_count):
                    logger.info(f"Processing page {page_num + 1} of {filename}")
                    page = doc[page_num]
                    text = page.get_text()
                    text_preview = text[:100].replace('\n', ' ') + "..." if len(text) > 100 else text.replace('\n', ' ')
                    logger.debug(f"Page {page_num + 1} text preview: {text_preview}")

                    documents.append({
                        "text": text,
                        "metadata": {
                            "file_name": filename,
                            "page_number": page_num + 1,
                        }
                    })
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                print(f"Error processing {filename}. See log for details.")

    if not found_files:
        logger.warning(f"No PDF files found in {directory_path}")
        print(f"No PDF files found in {directory_path}")
        print("Please add PDF files to continue.")
    else:
        logger.info(f"Loaded {len(documents)} document chunks total")
        print(f"Loaded {len(documents)} document chunks")

    return documents


# Text processing and metadata extraction
# TODO: Improve chunking strategy - maybe use recursive chunking
def process_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process documents with metadata and chunking"""
    logger.info(f"Processing {len(documents)} document chunks for indexing")
    processed_docs = []

    print("Processing documents...")

    for i, doc in enumerate(documents):
        logger.info(f"Processing document chunk {i + 1}/{len(documents)}")
        logger.info(f"Source: {doc['metadata']['file_name']}, Page: {doc['metadata']['page_number']}")

        # Simple sentence-based chunking
        # TODO: Try different chunking strategies
        sentences = doc["text"].split(". ")
        logger.info(f"Split text into {len(sentences)} sentences")

        # Create chunks of roughly 1024 characters with 128 character overlap
        chunks = []
        current_chunk = ""

        for j, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) < 1024:
                current_chunk += sentence + ". "
            else:
                # Save the current chunk
                chunks.append(current_chunk)
                # Start a new chunk with overlap
                overlap_point = max(0, len(current_chunk) - 128)
                current_chunk = current_chunk[overlap_point:] + sentence + ". "
                logger.debug(f"Created chunk {len(chunks)} with length {len(current_chunk)}")

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            logger.debug(f"Added final chunk with length {len(current_chunk)}")

        logger.info(f"Created {len(chunks)} chunks from document")

        # Process each chunk with metadata
        for k, chunk in enumerate(chunks):
            processed_doc = {
                "text": f"Metadata:\nfile_name:{doc['metadata']['file_name']}\npage_number:{doc['metadata']['page_number']}\nchunk:{k + 1}\n---\nContent:\n{chunk}",
                "metadata": {
                    **doc['metadata'],
                    "chunk": k + 1
                }
            }
            processed_docs.append(processed_doc)

            text_preview = chunk[:100].replace('\n', ' ') + "..." if len(chunk) > 100 else chunk.replace('\n', ' ')
            logger.debug(f"Chunk {k + 1} preview: {text_preview}")

    logger.info(f"Finished processing documents. Created {len(processed_docs)} total chunks.")
    return processed_docs


# Vector store setup function
# TODO: Try different vector stores (like FAISS)
def setup_vector_store(documents: List[Dict[str, Any]], embedding_model, collection_name: str = "guru_rag_collection"):
    """Set up ChromaDB vector store and add documents"""
    logger.info(f"Setting up vector store with collection name: {collection_name}")
    logger.info(f"Vector DB path: {VECTOR_DB_PATH}")

    # Initialize client, setting path to save data
    db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    logger.info("Initialized ChromaDB persistent client")

    try:
        # Check if collection exists
        collection = db.get_collection(collection_name)
        logger.info(f"Using existing collection '{collection_name}' with {collection.count()} documents")
        print(f"Using existing collection with {collection.count()} documents")
        return db, collection
    except Exception as e:
        # Create new collection
        logger.info(f"Collection '{collection_name}' not found: {e}")
        logger.info(f"Creating new collection '{collection_name}'")
        collection = db.create_collection(collection_name)
        print(f"Created new collection: {collection_name}")

    # Add documents to collection
    if documents:
        logger.info(f"Preparing to add {len(documents)} documents to vector store")
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        # Generate embeddings
        logger.info("Generating embeddings for documents...")
        print("Generating embeddings (this may take a while)...")
        start_time = time.time()
        embeddings = []

        for i, text in enumerate(texts):
            logger.info(f"Generating embedding for document {i + 1}/{len(texts)}")
            embedding = embedding_model.encode(text).tolist()
            embeddings.append(embedding)
            if (i + 1) % 10 == 0 or i + 1 == len(texts):
                logger.info(f"Progress: {i + 1}/{len(texts)} embeddings generated")

        elapsed_time = time.time() - start_time
        logger.info(f"Finished generating embeddings in {elapsed_time:.2f} seconds")

        # Add to collection
        logger.info("Adding documents to vector store...")
        print("Adding documents to vector store...")
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Added {len(documents)} documents to the vector store")
        print(f"Added {len(documents)} documents to the vector store")

    return db, collection


# RAG retrieval function
# TODO: Add filter capability by metadata
def retrieve_context(query: str, collection, embedding_model, top_k: int = 3):
    """Retrieve relevant documents for a query"""
    logger.info(f"Retrieving context for query: '{query}'")

    if not query.strip():
        logger.warning("Empty query provided")
        return "No query provided."

    if collection.count() == 0:
        logger.warning("Empty collection, no documents to search")
        return "No documents in the collection. Please add documents first."

    # Generate query embedding
    logger.info("Generating query embedding")
    start_time = time.time()
    query_embedding = embedding_model.encode(query).tolist()
    logger.info(f"Generated query embedding in {time.time() - start_time:.2f} seconds")

    # Query the collection
    logger.info(f"Querying collection for top {top_k} results")
    start_time = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )
    logger.info(f"Query completed in {time.time() - start_time:.2f} seconds")
    logger.info(
        f"Found {len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0} matching documents")

    # Check if we have results
    if not results["documents"] or not results["documents"][0]:
        logger.warning("No relevant documents found for query")
        return "No relevant documents found."

    # Format the retrieved documents for context
    context = ""
    for i, doc in enumerate(results["documents"][0]):
        logger.info(f"Including document {i + 1} in context")
        metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
        logger.info(f"Document metadata: {metadata}")
        doc_preview = doc[:200].replace('\n', ' ') + "..." if len(doc) > 200 else doc.replace('\n', ' ')
        logger.debug(f"Document content preview: {doc_preview}")

        context += f"Document {i + 1}:\n{doc}\n\n"

    logger.info(f"Prepared context with {len(context)} characters")
    return context


# Create custom reply function with Chain of Thought for AutoGen agents
# TODO: Add support for multiple thinking steps and revision
def custom_reply_function(agent, messages=None, sender=None, config=None):
    """Custom reply function that uses our Ollama LLM with chain-of-thought reasoning"""
    # We need to access the local_llm object from main
    global local_llm

    logger.info(f"Custom reply function called for agent '{agent.name}' from sender '{sender}'")
    logger.info(f"Received {len(messages) if messages else 0} messages")

    # Get the latest message
    if messages and len(messages) > 0:
        last_message = messages[-1]
        content = last_message.get("content", "")
        content_preview = content[:200].replace('\n', ' ') + "..." if len(content) > 200 else content.replace('\n', ' ')
        logger.info(f"Processing message: {content_preview}")

        # Add chain-of-thought instructions if not already there
        if "chain of thought" not in content.lower() and "[THINKING]" not in content:
            logger.info("Adding chain-of-thought instructions to prompt")
            content += "\n\nThink step by step before answering. Show your reasoning process labeled as [THINKING], then provide your final answer labeled as [ANSWER]."

        # Call Ollama LLM
        logger.info("Calling Ollama LLM for response")
        print("Thinking...")
        start_time = time.time()
        response = local_llm([{"role": "user", "content": content}])
        elapsed_time = time.time() - start_time
        logger.info(f"Ollama LLM completed response in {elapsed_time:.2f} seconds")

        # Full response with CoT - only log to file, not console
        full_response = response.get("content", "")

        # Log the full response to the file
        logger.info(f"Full response length: {len(full_response)}")
        logger.info("=== CHAIN OF THOUGHT REASONING ===")
        logger.info(full_response)
        logger.info("=== END OF CHAIN OF THOUGHT ===")

        return full_response

    logger.warning("No messages provided to custom_reply_function")
    return "I don't have enough context to respond."


# Main execution
# TODO: Add command line arguments
def main():
    global local_llm  # Make the LLM accessible to the custom reply function

    logger.info("Starting my GURU RAG System with Chain of Thought")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data directory: {DATA_DIR}")

    print("\n" + "=" * 50)
    print("MY GURU RAG SYSTEM WITH CHAIN OF THOUGHT")
    print("=" * 50)
    print(f"Log file: {log_filename}")
    print("All detailed steps are being recorded in the log file.\n")

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        logger.info(f"Creating data directory at {DATA_DIR}")
        print(f"Creating data directory at {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.warning("No PDF files found. Please add PDFs to the data directory")
        print(f"Please add PDF files to {DATA_DIR} before running again.")
        return

    # 0. Set up models
    logger.info("Setting up models")
    embedding_model = setup_embedding_model()
    local_llm = OllamaLLM()

    # 1. Extract - Load documents
    # TODO: Add recursive directory scanning
    logger.info("Loading documents from data directory")
    docs = load_documents(DATA_DIR)

    if not docs:
        logger.warning("No documents loaded")
        print("No documents loaded. Please add PDF files to the data directory.")
        return

    # 2. Transform - Process documents
    logger.info("Processing documents")
    processed_docs = process_documents(docs)

    # 3. Index - Create vector store
    logger.info("Creating/loading vector store")
    db, collection = setup_vector_store(processed_docs, embedding_model)

    # 4. Set up AutoGen agents (version 0.1.14 compatible)
    # TODO: Upgrade to newer autogen version
    logger.info("Setting up AutoGen agents")

    # Create the assistant agent with CoT system message
    system_message = """You are the GURU assistant. You help users find information in their documents. 

When answering questions:
1. Think step by step about the information provided in the documents.
2. Show your detailed chain of thought reasoning.
3. Label your thinking process with [THINKING].
4. After your analysis, provide a concise and clear answer labeled with [ANSWER].
5. Make sure to cite the source documents when providing information.

Your goal is to be both thorough in your reasoning and accurate in your final response."""

    logger.info("Creating assistant agent with system message")
    logger.info(f"System message: {system_message}")

    assistant = AssistantAgent(
        name="guru_assistant",
        system_message=system_message,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        llm_config=False,  # Disable default LLM
        human_input_mode="NEVER"
    )
    logger.info("Assistant agent created")

    # Override the default reply method
    assistant._reply_func = custom_reply_function
    logger.info("Overrode default reply function with custom implementation")

    # Create the user proxy agent
    logger.info("Creating user proxy agent")
    user_proxy = UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="ALWAYS",
        code_execution_config={"use_docker": False, "work_dir": WORKSPACE_DIR}
    )
    logger.info("User proxy agent created")

    # Start the conversation
    logger.info("Starting conversation loop")
    print("\nGURU system initialized. Type your questions or 'exit' to quit.")
    print("(All processing steps and chain-of-thought are logged to file)\n")

    # Conversation loop
    print("GURU: What would you like to know about your documents?")
    logger.info("GURU system is ready for user input")

    conversation_counter = 0

    while True:
        conversation_counter += 1
        user_query = input("\nYou: ").strip()
        logger.info(f"Conversation turn {conversation_counter}")
        logger.info(f"User input: '{user_query}'")

        if user_query.lower() == 'exit':
            logger.info("User requested exit")
            print("\nExiting GURU system. Goodbye!")
            logger.info("GURU system shutting down")
            break

        # Get context from RAG
        print("Retrieving information...")
        logger.info("Retrieving relevant context for query")
        start_time = time.time()
        context = retrieve_context(user_query, collection, embedding_model)
        elapsed_time = time.time() - start_time
        logger.info(f"Context retrieval completed in {elapsed_time:.2f} seconds")
        logger.info(f"Retrieved context length: {len(context)}")

        # Prepare prompt with context and query, including chain-of-thought instructions
        # TODO: Add customizable prompts
        logger.info("Preparing augmented query with context and CoT instructions")
        augmented_query = f"""Use the following retrieved information to answer the question.

Retrieved information:
{context}

Question: {user_query}

Instructions:
1. First, think step by step about what the question is asking.
2. Then, analyze the retrieved information and identify relevant details.
3. Show your reasoning process labeled as [THINKING].
4. Provide your final answer labeled as [ANSWER].

Start your analysis now:"""

        logger.info(f"Augmented query length: {len(augmented_query)}")
        logger.debug(f"Full augmented query: {augmented_query}")

        # Create a chat message
        logger.info("Creating chat message for assistant")
        chat_message = {"role": "user", "content": augmented_query}

        # Generate a reply using our custom function
        logger.info("Generating reply from assistant")
        start_time = time.time()
        response = custom_reply_function(assistant, [chat_message])
        elapsed_time = time.time() - start_time
        logger.info(f"Assistant generated response in {elapsed_time:.2f} seconds")
        logger.info(f"Response length: {len(response)}")

        # Print only the final answer to the console
        # TODO: Add option to show full reasoning
        if "[ANSWER]" in response:
            logger.info("Extracting final answer from response")
            final_answer = response.split("[ANSWER]")[1].strip()
            logger.info(f"Final answer: {final_answer}")
            print(f"\nGURU: {final_answer}")
        else:
            logger.warning("No [ANSWER] tag found in response")
            # For responses without an [ANSWER] tag, still print a clean version
            # Remove [THINKING] sections if present
            if "[THINKING]" in response:
                clean_response = response.split("[THINKING]")[0].strip()
                logger.info("Using cleaned response as answer")
                print(f"\nGURU: {clean_response}")
            else:
                logger.info("Using full response as answer")
                print(f"\nGURU: {response}")

        logger.info("Completed conversation turn, waiting for next user input")
        # TODO: Add conversation history saving


if __name__ == "__main__":
    try:
        logger.info("--------- STARTING guru-ag-rag-poc RAG APPLICATION ---------")
        main()
        logger.info("--------- MY guru-ag-rag-poc RAG APPLICATION COMPLETED SUCCESSFULLY ---------")
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nExiting guru-autogen-rag-poc system. Goodbye!")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        print(f"See the log file for details: {log_filename}")