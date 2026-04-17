from imageprocessor import ImageProcessor
from chromadbprocess import ChromaDBProcessor
from worddocprocessor import WordDocumentProcessor
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

def Test1():
    """Main entry point for the RAG image processor."""
    # Configuration for ImageProcessor
    api_key = os.getenv("OPENAI_API_KEY")  # Replace with your actual OpenAI API key
    folder_path = "./my_photos"   # Replace with your image folder path
    
    # Initialize and use ImageProcessor
    print("Initializing ImageProcessor...")
    image_processor = ImageProcessor(api_key)
    # image_processor.process_folder(folder_path)  # Uncomment to run
    
    # Initialize and use ChromaDBProcessor
    print("Initializing ChromaDBProcessor...")
    chroma_processor = ChromaDBProcessor()
    chroma_processor.setup()  # Uncomment to setup ChromaDB (warning: ~700 MB download)
    # Uncomment to download and setup embeddings (warning: ~700 MB download)
    article_df = chroma_processor.setup()
    print(article_df.head())
    chroma_processor.read_dataframe()
    print("Setup complete!")

def main():
    """Main entry point for the RAG word document processor."""
    # Configuration for WordDocumentProcessor
    api_key = os.getenv("OPENAI_API_KEY")  # Replace with your actual OpenAI API key
    file_path = r"C:\work\Python\ragimage\ragimage\src\doctest.docx"  # Replace with your Word document path
    
    # Initialize and use WordDocumentProcessor
    print("You can double-click")
    word_processor = WordDocumentProcessor()
    word_processor.embed_and_store(file_path)  # Process and store the document
    query_text = "d"  # Example query
    word_processor.query(query_text)  # Query the stored document
    
if __name__ == "__main__":
    main()