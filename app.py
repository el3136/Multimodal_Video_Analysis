# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import time
from helpers import *

# Set page configuration
st.set_page_config(
    page_title = "Video_Analyzer",
    page_icon = '🎥',
    layout = "wide"
)

# Apply custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Function to download the conversation chat
def download_conversation(history):
    """Converts chat history to a string and returns it as a downloadable file."""
    text_history = "\n".join([f"{item['role']}: {item['parts'][0]}" for item in history])
    return text_history.encode(), "conversation_history.txt"

# Main function
def main():

    # Initialize or get session state variables
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""

    if "summarized" not in st.session_state:
        st.session_state.summarized = False

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False

    if "documents_added" not in st.session_state:
        st.session_state.documents_added = False

    # Main Title and Subtitle
    st.markdown('<div class="main-title">Welcome to VideoMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">"Turning Monologues into Conversations"</div>', unsafe_allow_html=True)
    st.divider()

    # This is the prompt to the LLM for summarizing the transcript
    prompt = "I have a transcription of an audio recording. Generate a brief documentation from it. The documentation should include the following sections:\
                    - Introduction: A detailed overview of the content.\
                    - Key points: The main ideas or arguments presented in the audio.Include the timestamp \
                        where each key point was discussed. The transcript is provided with timestamps \
                        in [MM:SS] format.\
                    - Conclusion: A brief conclusion or summary of the key takeaways."

    # Sidebar for entering the YouTube URL
    with st.sidebar:
        st.markdown("### About VideoMind")
        st.markdown("""
        VideoMind transforms YouTube videos into interactive conversations. 
        Paste a URL and chat about your video content to get summaries and answer questions.
        
        ## How It Works:

        1. Paste any YouTube URL.
        2. AI processes the video transcript.
        3. Get an immediate summary of the video which is downloadable in text file.
        4. Start chatting with the content - ask anything about the video.
        5. Receive intelligent responses with relevant timestamps and also download the whole chat history in a text file.
        """)

        st.markdown("Enter your Google API key below to get started.")
        google_api_key = st.text_input(label = "Google API key", placeholder="Ex AIxxxxxxxxxxxxxxxx",
            key ="google_api_key_input", type= 'password', help = "How to get a Google api key: Visit https://ai.google.dev/gemini-api/docs to know more.")
        
        if google_api_key:
            # Initialize the genai client
            client = genai.Client(api_key=google_api_key)
            st.session_state["api_key"] = google_api_key
            try:
                # Store the client in session state
                st.session_state["genai_client"] = client
                st.success("API Key validated and client initialized!")
            except Exception as e:
                st.error(f"Error initializing client: {e}")
        
        youtube_url = st.text_input("Enter a valid Youtube URL", help="A valid Youtube URL eg: https://www.youtube.com/watch?v=0ctat6RBrFo")

    # Add a footer
    st.sidebar.markdown("Built with Streamlit and Google Gemini")

    if "genai_client" in st.session_state:
        if youtube_url:
            try:
                # Initialize the YouTubeProcessor
                yt_processor = YouTubeProcessor(video_url=youtube_url)
                transcript_with_timestamps = yt_processor.transcribe()
                transcript = yt_processor.get_formatted_transcript()
                transcript_with_timestamps_str = ' '.join([f"[{item['start']:.3f}] {item['text']}" for item in transcript_with_timestamps])
                # Getting the transcipt of the video
                transcript_button = st.sidebar.button("Get the transcript of the video", key="get_transcript")
                if transcript_button:
                    yt_processor.display_thumbnail()  #Displays the thumbnail
                    
                    with st.container():
                        st.markdown(transcript)
                        st.session_state.transcript = str(transcript)
            except Exception as e:
                st.info(e)

            # Implementation of Summarizer
            summarize = st.sidebar.button("Summarize the video", key="summarize_video")
            if summarize:
                with st.spinner("Processing..."):
                    
                    summarizer = DocumentSummariser(prompt)
                    summary_doc = summarizer.summarize_doc(transcript_data=transcript_with_timestamps, google_api_key=google_api_key)
                    with st.container():
                        st.markdown(summary_doc)
                        st.session_state.summarized = True
                st.download_button(label="Download the summary", data=summary_doc, file_name="video_summary.txt")

            # Implementation of Video Chatbot
                    
            # Only initialize the chatbot once when transcript is available
            if transcript_with_timestamps_str and not st.session_state.chatbot_initialized:
                chat_button = st.sidebar.button("Ask questions about the video", key="chat")
                if chat_button:
                    with st.spinner("Initializing VideoMind"):
                        # Initialize the Video Chatbot
                        st.session_state.chatbot = VideoChatbot(google_api_key=google_api_key, db_name="qa_database")
                        
                        # Add documents to the database
                        documents = [transcript_with_timestamps_str]
                        st.session_state.chatbot.add_documents(documents)
                        st.session_state.chatbot_initialized = True
                        st.success("VideoMind initialized successfully!")

            # Display chat interface only when chatbot is initialized
            if st.session_state.chatbot_initialized:
                # Display welcome message only once
                if not st.session_state.conversation_history:
                    welcome_message = "Hello! Welcome to VideoMind, I am your Video assistant. Ask me any questions related to the Youtube video."
                    st.session_state.conversation_history.append({"role": "assistant", "parts": [welcome_message]})
                
                # Display the conversation history
                for turn in st.session_state.conversation_history:
                    role = turn["role"]
                    content = turn["parts"][0]
                    with st.chat_message("user" if role == "user" else "assistant"):
                        st.markdown(content)
                
                # Chat input area
                user_input = st.chat_input("Ask your question here...")
                
                if user_input:
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    
                    # Get chatbot response
                    with st.spinner("Thinking..."):
                        result = st.session_state.chatbot.query(user_input)
                        assistant_reply = result["answer"]
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        def stream_answer():
                            for word in assistant_reply.split(" "):
                                yield word + " "
                                time.sleep(0.05)
                        st.write_stream(stream_answer)
                    
                    # Note: The query method already updates the chatbot's internal conversation history
                    # We only need to update our display history if it's separate
                    if st.session_state.conversation_history[-1]["role"] != "user" or st.session_state.conversation_history[-1]["parts"][0] != user_input:
                        st.session_state.conversation_history.append({"role": "user", "parts": [user_input]})
                    if st.session_state.conversation_history[-1]["role"] != "assistant" or st.session_state.conversation_history[-1]["parts"][0] != assistant_reply:
                        st.session_state.conversation_history.append({"role": "assistant", "parts": [assistant_reply]})
                
                # Clear conversation button
                if st.button("🧹 Clear Conversation", key="clear_history"):
                    st.session_state.chatbot.clear_history()
                    st.session_state.conversation_history = []
                    st.rerun()
                
                # Download the chat button
                if len(st.session_state.conversation_history) > 1:
                    text_to_download, filename = download_conversation(st.session_state.conversation_history)
                    st.download_button(
                        label="Download Chat",
                        data=text_to_download,
                        file_name=filename,
                        mime="text/plain",
                    )
                else:
                    st.info("No conversation history yet.")
            else:
                if transcript_with_timestamps_str:
                    st.sidebar.info("Click on 'Ask questions about the video' to start chat.")
                else:
                    st.sidebar.warning("No transcript available. Please load a video first.")

if __name__=="__main__":
    main()