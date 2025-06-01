from google import genai
from google.genai import types
from pydantic import BaseModel, HttpUrl, field_validator
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict, Any, Optional
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import streamlit as st

# 3 Classes:
# YouTubeProcessor: This is for extracting the transcript of the video
# DocumentSummarizer: This is for generating a summary of the video along with timestamps using gemini-2.0-flash model from Google.
# VideoChatbot: This is for implementing a chatbot with which the user can interact with in QA format. 

# RAG implementation: The video transcript is embedded using the embedding model and are stored in Chroma DB.
# The LLM retrieves the relevant content from the database based on the user query to answer the questions effectively.

class YouTubeProcessor(BaseModel):
    video_url: HttpUrl
    video_id: str = None
    
    # Custom validator for video_url to extract the 'v' parameter (YouTube video ID)
    @field_validator('video_url')
    def validate_video_url(cls, value):
        # Convert the HttpUrl object to a string to parse it with urlparse
        value_str = str(value)
        
        # Parse the URL using urllib
        parsed_url = urlparse(value_str)
        
        # Extract query parameters
        query_params = parse_qs(parsed_url.query)
        
        # Check if the 'v' parameter exists
        if 'v' not in query_params:
            raise ValueError("Invalid YouTube video URL format. Must contain 'v='")
            
        # Return the original value, not the video_id
        return value
    
    # Model initialization to set video_id
    def model_post_init(self, __context):
        # Extract video_id from URL
        parsed_url = urlparse(str(self.video_url))
        query_params = parse_qs(parsed_url.query)
        if 'v' in query_params:
            self.video_id = query_params['v'][0]
    
    # Function to extract the transcript of the video
    def transcribe(self):
        """Extract transcript with timestamps from the YouTube video."""
        if not self.video_id:
            raise ValueError("Video ID not available")
            
        try:
            # Return the raw transcript data instead of joining it into a string
            return YouTubeTranscriptApi.get_transcript(self.video_id)
        except Exception as e:
            raise e
            
    def get_formatted_transcript(self):
        """Get transcript as a plain text string."""
        transcript_data = self.transcribe()
        full_transcript = ""
        for item in transcript_data:
            full_transcript += item["text"] + " "
        return full_transcript
    
    # Function to display the thumbnail of the YouTube video
    def display_thumbnail(self):
        # If video ID is extracted, generate the thumbnail URL
        if self.video_id:
            thumbnail_url = f"https://img.youtube.com/vi/{self.video_id}/maxresdefault.jpg"
            # Display the thumbnail image in the notebook
            st.image(thumbnail_url)
        else:
            print("Invalid YouTube URL")

class DocumentSummariser:
    def __init__(self, prompt, client=None):
        self.prompt = prompt
        self.client = client
    
    def summarize_doc(self, transcript_data, google_api_key):
        """Execute the prompt on the transcript with timestamp data.
        
        Args:
            transcript_data (list): The transcript segments with timestamps
            
        Returns:
            str: The generated summary text with timestamps
        """
        # Process transcript to include timestamps in a format the LLM can understand
        transcript_with_timestamps = self.format_transcript_with_timestamps(transcript_data)
        
        # Initialize client if needed...
        if not self.client:
            self.client = genai.Client(api_key=google_api_key)
        
        config = types.GenerateContentConfig(temperature=0.0)
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                config=config,
                contents=[self.prompt, transcript_with_timestamps],
            )
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return None
    
    def format_transcript_with_timestamps(self, transcript_data):
        """Format transcript data with readable timestamps for the LLM."""
        formatted_text = ""
        for item in transcript_data:
            # Convert seconds to MM:SS format
            minutes = int(item['start'] // 60)
            seconds = int(item['start'] % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            
            # Add timestamp and text
            formatted_text += f"[{timestamp}] {item['text']}\n"
        
        return formatted_text
    
# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True
    def __init__(self, client):
        self.client = client

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]
    
class VideoChatbot:
    def __init__(self, google_api_key, db_name: str = "qa_database"):
        """
        Initialize the Video Chatbot with a database and embedding function.
        
        Args:
            client: The Google Generative AI client
            db_name: Name for the ChromaDB collection
        """
        self.client = genai.Client(api_key=google_api_key)
        self.db_name = db_name

        # Initialize embedding function
        self.embed_fn = GeminiEmbeddingFunction(self.client)
        self.embed_fn.document_mode = True

        # Initialize ChromaDB
        persist_directory = "./chroma_db"  # Replace with your desired directory
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.db = self.chroma_client.get_or_create_collection(
            name = self.db_name,
            embedding_function = self.embed_fn
        )

        # Initialize conversation history
        self.conversation_history = []
        
    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None):
        """
        Add documents to the database.
        
        Args:
            documents: List of document texts to add
            ids: Optional list of IDs for the documents. If not provided, 
                 sequential numbers will be used
        """

        if ids is None:
            ids = [str(i) for i in range(len(documents))]

        # Set the document mode for adding documents
        self.embed_fn.document_mode = True

        # Add documents to the database
        self.db.add(documents=documents, ids = ids)

        return f"Added {len(documents)} documents to the database."

    
    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Query the database and generate an answer using the LLM.
        
        Args:
            query_text: The query text
            n_results: Number of relevant passages to retrieve
            
        Returns:
            A dictionary containing the answer and retrieved passages
        """
        # Check if the user is trying to end the conversation
        exit_phrases = ["thank you", "thanks", "thanks!", "thank you!", "thank you very much"]
        if query_text.strip().lower() in exit_phrases:
            farewell_message = "You're very welcome! If you have more questions later, just ask 😊"
            self.conversation_history.append({"role": "user", "parts": [query_text]})
            self.conversation_history.append({"role": "model", "parts": [farewell_message]})
            return {
                "answer": farewell_message,
                "retrieved_passages": [],
                "prompt": "",
                "history": self.conversation_history
            }
            
        # Switch to query mode for generating embeddings
        self.embed_fn.document_mode = False
        
        # Search the Chroma DB using the specified query
        result = self.db.query(query_texts=[query_text], n_results=n_results)
        retrieved_passages = result["documents"][0]

        # Construct prompt with query and retrieved passages
        prompt = self.construct_prompt(query_text, retrieved_passages)

        config = types.GenerateContentConfig(temperature=0.0)

        # Generate answer using the LLM
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            config = config,
            contents=prompt
        )

        answer_text = response.text

        # Update conversation history
        self.conversation_history.append({"role": "user", "parts": [query_text]})
        self.conversation_history.append({"role": "model", "parts": [answer_text]})
        
        return {
            "answer": answer_text,
            "retrieved_passages": retrieved_passages,
            "prompt": prompt,
            "history": self.conversation_history
        }

    def construct_prompt(self, query: str, retrieved_passages: List[str], history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Construct a prompt for the LLM using the query and retrieved passages.
        
        Args:
            query: The user's question
            retrieved_passages: Passages retrieved from the database
            history: Optional list of previous conversation turns

        Returns:
            A formatted prompt string
        """
        query_oneline = query.replace("\n", " ")

        # This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
        prompt = f"""You are a helpful and friendly assistant that answers questions based on the transcribed text from a video. Your responses should be clear, comprehensive, and accessible to a non-technical audience. When answering, make sure to:

        Provide complete sentences and explain the relevant background information from the video.
        
        Include specific timestamps when referring to certain details from the video, so the user can easily locate the information.
        
        If the question is irrelevant to the content of the video, kindly explain that the question is not applicable.
        
        Break down complicated concepts in a simple and conversational way, avoiding jargon or overly technical terms.
        
        Your goal is to help the user understand the video content, offering detailed yet approachable explanations.
        
        """

        # Add conversation history to the prompt
        if history:
            prompt += "Here is the previous conversation for context:\n"
            for turn in history:
                role = turn["role"].capitalize()
                content = turn["parts"][0].replace("\n"," ")
                prompt += f"{role}: {content}\n"
            prompt += "\n"

        prompt += f"QUESTION: {query_oneline}\n"
        
        # Add the retrieved documents to the prompt.
        for i, passage in enumerate(retrieved_passages):
            passage_oneline = passage.replace("\n", " ")
            prompt += f"PASSAGE {i+1}: {passage_oneline}\n"

        prompt += "ANSWER:"
        
        return prompt

    def clear_history(self):
        """
        Clears the conversation history
        """
        self.conversation_history = []
        print("Convesation history cleared.")
    
    def display_answer(self, result = Dict[str,Any]) -> None:
        """
        Display the answer using Markdown.
        
        Args:
            result: Dictionary containing the answer and retrieved passages
        """
        print(result.get("answer"))

    def display_passages(self, result: Dict[str, Any]) -> None:
        """
        Display the retrieved passages based on the query

        Args:
            result: Dictionary containing the answer and retrieved passages
        """
        print("Retrieved Passages:")
        for i, passage in enumerate(result["retrieved_passages"]):
            print(f"\nPassage {i+1}:")
            print(passage)