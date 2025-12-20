"""
ARSLM Frontend - Streamlit Application

Interactive web interface for ARSLM chatbot.
"""

import streamlit as st
import requests
from datetime import datetime
import uuid
import os
from typing import List, Dict
import json

# Page configuration
st.set_page_config(
    page_title="ARSLM Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")


# Custom CSS
def load_css():
    """Load custom CSS styles."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .message-role {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .message-content {
        line-height: 1.6;
    }
    .message-time {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session():
    """Initialize session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'api_available' not in st.session_state:
        st.session_state.api_available = check_api_health()


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def send_message(message: str, params: Dict) -> Dict:
    """
    Send message to API and get response.
    
    Args:
        message: User message
        params: Generation parameters
        
    Returns:
        API response
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                **params
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with API: {str(e)}")
        return None


def get_conversation_history() -> List[Dict]:
    """Get conversation history from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/history/{st.session_state.session_id}",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()['messages']
        return []
    except:
        return []


def clear_conversation_history():
    """Clear conversation history."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/history/{st.session_state.session_id}",
            timeout=5
        )
        if response.status_code == 200:
            st.session_state.messages = []
            st.success("Conversation history cleared!")
        else:
            st.error("Failed to clear history")
    except Exception as e:
        st.error(f"Error clearing history: {str(e)}")


def display_message(role: str, content: str, timestamp: str = None):
    """Display a chat message."""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div class="message-role">{icon} {role.capitalize()}</div>
        <div class="message-content">{content}</div>
        <div class="message-time">{timestamp}</div>
    </div>
    """, unsafe_allow_html=True)


def sidebar():
    """Render sidebar with settings and info."""
    with st.sidebar:
        st.markdown('<h2 style="text-align: center;">⚙️ Settings</h2>', unsafe_allow_html=True)
        
        # API Status
        with st.expander("📊 API Status", expanded=True):
            if st.session_state.api_available:
                st.success("✅ API is online")
            else:
                st.error("❌ API is offline")
                if st.button("🔄 Retry Connection"):
                    st.session_state.api_available = check_api_health()
                    st.rerun()
        
        # Generation Parameters
        with st.expander("🎛️ Generation Parameters"):
            max_length = st.slider(
                "Max Length",
                min_value=10,
                max_value=500,
                value=100,
                help="Maximum length of generated response"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Higher = more creative, Lower = more deterministic"
            )
            
            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=100,
                value=50,
                help="Number of top tokens to consider"
            )
            
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Nucleus sampling threshold"
            )
        
        # Session Info
        with st.expander("📝 Session Info"):
            st.markdown(f"""
            <div class="sidebar-info">
                <strong>Session ID:</strong><br>
                <code>{st.session_state.session_id[:8]}...</code><br><br>
                <strong>Messages:</strong> {len(st.session_state.messages)}<br>
                <strong>Status:</strong> Active
            </div>
            """, unsafe_allow_html=True)
        
        # Actions
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_conversation_history()
                st.rerun()
        
        with col2:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()
        
        # Export History
        if st.button("💾 Export History", use_container_width=True):
            history = get_conversation_history()
            if history:
                json_str = json.dumps(history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"conversation_{st.session_state.session_id[:8]}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **ARSLM v1.0.0**
            
            Adaptive Reasoning Semantic Language Model
            
            - 🧠 Intelligent conversation
            - 🔒 Privacy-focused
            - ⚡ Fast inference
            - 🌍 Global deployment
            
            [GitHub](https://github.com/benjaminpolydeq/ARSLM) | [Docs](#)
            """)
        
        return {
            'max_length': max_length,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        }


def main():
    """Main application."""
    # Load CSS
    load_css()
    
    # Initialize session
    initialize_session()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 ARSLM Chat</h1>', unsafe_allow_html=True)
    
    # Sidebar with settings
    params = sidebar()
    
    # Check API availability
    if not st.session_state.api_available:
        st.warning("⚠️ API is not available. Please check your connection or start the API server.")
        st.info("Run: `uvicorn api.main:app --reload`")
        st.stop()
    
    # Chat interface
    chat_container = st.container()
    
    # Display conversation history
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.info("👋 Welcome! Start a conversation by typing a message below.")
        else:
            for message in st.session_state.messages:
                display_message(
                    role=message['role'],
                    content=message['content'],
                    timestamp=message.get('timestamp', '')
                )
    
    # Input area
    st.markdown("---")
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Your message",
            key="user_input",
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True, type="primary")
    
    # Handle message sending
    if send_button and user_input:
        # Add user message to display
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Show loading spinner
        with st.spinner("🤔 Thinking..."):
            # Send to API
            response = send_message(user_input, params)
            
            if response:
                # Add assistant response
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response['response'],
                    'timestamp': response.get('timestamp', datetime.now().strftime("%H:%M:%S"))
                })
        
        # Rerun to update UI
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Made with ❤️ by Benjamin Amaad Kama | 
        <a href="https://github.com/benjaminpolydeq/ARSLM" target="_blank">GitHub</a> | 
        <a href="#" target="_blank">Documentation</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
