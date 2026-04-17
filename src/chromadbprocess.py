import openai
import os
import pandas as pd
import wget
from ast import literal_eval
import zipfile
import warnings

# Chroma's client library for Python
import chromadb

EMBEDDING_MODEL = "text-embedding-3-small"
# Ignore unclosed SSL socket warnings - optional in case you get these errors
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ChromaDBProcessor:
    """Handles ChromaDB vector database setup and Wikipedia embeddings processing."""
    
    # I've set this to our new embeddings model, this can be changed to the embedding model of your choice
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Loading the wikipedia articles that have already been embedded using the text-embedding-3-small model
    EMBEDDINGS_URL = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
    ZIP_FILE = "vector_database_wikipedia_articles_embedded.zip"
    DATA_DIR = "../data"
    CSV_FILE = '../data/vector_database_wikipedia_articles_embedded.csv'
    
    def __init__(self):
        """Initialize the ChromaDB processor."""
        self.article_df = None
    
    def download_embeddings(self):
        """Downloads the pre-embedded Wikipedia articles from OpenAI CDN.
        
        The file is ~700 MB so this will take some time.
        """
        print(f"Downloading embeddings from {self.EMBEDDINGS_URL}...")
        #wget.download(self.EMBEDDINGS_URL)
        print("\nDownload complete.")
    
    def extract_embeddings(self):
        """Extracts the downloaded zip file to the data directory."""
        print(f"Extracting {self.ZIP_FILE}...")
        with zipfile.ZipFile(self.ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(self.DATA_DIR)
        print("Extraction complete.")
    
    def load_embeddings(self):
        """Loads the embeddings CSV file into a pandas DataFrame."""
        if not os.path.exists(self.CSV_FILE):
            raise FileNotFoundError(f"CSV file not found at {self.CSV_FILE}. Please run download and extract first.")
        
        print(f"Loading embeddings from {self.CSV_FILE}...")
        self.article_df = pd.read_csv(self.CSV_FILE)
        print(f"Loaded {len(self.article_df)} articles.")
        return self.article_df
    
    def setup(self):
        """Complete setup process: download, extract, and load embeddings."""
        self.download_embeddings()
        self.extract_embeddings()
        return self.load_embeddings()
    
    def get_dataframe(self):
        """Returns the loaded dataframe."""
        if self.article_df is None:
            raise ValueError("Dataframe not loaded. Call load_embeddings() or setup() first.")
        return self.article_df
    def read_dataframe(self):
        """Reads vectors from strings back into a list."""
        self.article_df['title_vector'] = self.article_df.title_vector.apply(literal_eval)
        self.article_df['content_vector'] = self.article_df.content_vector.apply(literal_eval)
        # Set vector_id to be a string
        self.article_df['vector_id'] = self.article_df['vector_id'].apply(str)
        self.article_df.info(show_counts=True)
    def initialize_chromadb(self):
        """Initializes the ChromaDB client and collection."""
        self.chroma_client = chromadb.EphemeralClient() # Equivalent to chromadb.Client(), ephemeral.  
        # Uncomment for persistent client
        # self.chroma_client = chromadb.PersistentClient()
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  
        # Test that your OpenAI API key is correctly set as an environment variable
        # Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.

        # Note. alternatively you can set a temporary env variable like this:
        # os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

        if os.getenv("OPENAI_API_KEY") is not None:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            print ("OPENAI_API_KEY is ready")
        else:
            print ("OPENAI_API_KEY environment variable not found")

        embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)
        self.wikipedia_content_collection = self.chroma_client.create_collection(name='wikipedia_content', embedding_function=embedding_function)
        self.wikipedia_title_collection = self.chroma_client.create_collection(name='wikipedia_titles', embedding_function=embedding_function)
        # Add the content vectors
        self.wikipedia_content_collection.add(
                ids=self.article_df.vector_id.tolist(),
                embeddings=self.article_df.content_vector.tolist(),
    )

        # Add the title vectors
        self.wikipedia_title_collection.add(
            ids=self.article_df.vector_id.tolist(),
            embeddings=self.article_df.title_vector.tolist(),
        )
    def query_collection(self, collection, query, max_results, dataframe):
        results = collection.query(query_texts=query, n_results=max_results, include=['distances']) 
        df = pd.DataFrame({
                'id':results['ids'][0], 
                'score':results['distances'][0],
                'title': dataframe[dataframe.vector_id.isin(results['ids'][0])]['title'],
                'content': dataframe[dataframe.vector_id.isin(results['ids'][0])]['text'],
                })
    
        return df
    def query_header(self):
        title_query_result = self.query_collection(
        collection=self.wikipedia_title_collection,
        query="modern art in Europe",
        max_results=10,
        dataframe=self.article_df
        )
        return title_query_result.head()
    def query_content(self):
        content_query_result = self.query_collection(
            collection=self.wikipedia_content_collection,
            query="Famous battles in Scottish history",
            max_results=10,
            dataframe=self.article_df
        )
        return content_query_result.head()