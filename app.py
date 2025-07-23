import streamlit as st
import json
import requests
from datetime import datetime
import websocket
import threading
import queue
import time
from typing import Dict, Any, Optional

# Page config
st.set_page_config(
    page_title="Agentic Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
    
if 'conversations' not in st.session_state:
    st.session_state.conversations = []
    
if 'is_typing' not in st.session_state:
    st.session_state.is_typing = False

if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False

if 'last_message_time' not in st.session_state:
    st.session_state.last_message_time = 0

if 'message_count' not in st.session_state:
    st.session_state.message_count = 0

if 'processing_message' not in st.session_state:
    st.session_state.processing_message = False

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# API endpoints
BASE_URL = "http://localhost:8001"
WS_URL = "ws://localhost:8001"

# Global WebSocket state (outside session state for thread safety)
class WebSocketManager:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.url = None
        self.ws = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        with self.lock:
            return self.ws is not None and hasattr(self.ws, 'connected') and self.ws.connected

    def connect(self, conversation_id: str):
        """Connect to WebSocket server in a background thread."""
        if self.thread and self.thread.is_alive():
            self.disconnect()

        self.url = f"{WS_URL}/ws/{conversation_id}"
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.thread.start()

    def _connection_loop(self):
        """The main loop for connecting and listening to the WebSocket."""
        max_attempts = 3
        attempt = 0
        while not self.stop_event.is_set() and attempt < max_attempts:
            try:
                attempt += 1
                with self.lock:
                    self.ws = websocket.create_connection(self.url, timeout=10)
                
                self.message_queue.put(('status', 'WebSocket connected'))
                attempt = 0  # Reset on successful connection
                self._listen_for_messages()

            except Exception as e:
                if self.stop_event.is_set():
                    break
                error_msg = f"Connection failed (attempt {attempt}): {str(e)}"
                self.message_queue.put(('error', error_msg))
                if attempt < max_attempts:
                    time.sleep(2)  # Wait before retrying
            finally:
                with self.lock:
                    if self.ws:
                        try:
                            self.ws.close()
                        except Exception:
                            pass
                        self.ws = None
                if not self.stop_event.is_set() and attempt >= max_attempts:
                    self.message_queue.put(('status', 'WebSocket disconnected'))

    def _listen_for_messages(self):
        """Listen for incoming messages on the WebSocket."""
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    if not self.ws:
                        break
                try:
                    message = self.ws.recv()
                    if message:
                        data = json.loads(message)
                        self.message_queue.put(('message', data))
                except websocket.WebSocketTimeoutException:
                    continue  # Timeout is normal, keep listening
        except (websocket.WebSocketConnectionClosedException, ConnectionResetError):
            if not self.stop_event.is_set():
                self.message_queue.put(('error', 'Connection closed unexpectedly.'))
        except Exception as e:
            if not self.stop_event.is_set():
                self.message_queue.put(('error', f'An error occurred: {str(e)}'))

    def disconnect(self):
        """Disconnect WebSocket and stop the background thread."""
        self.stop_event.set()
        with self.lock:
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass
                self.ws = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.thread = None

    def send_message(self, message_data):
        """Send message via WebSocket."""
        if not self.is_connected():
            self.message_queue.put(('error', 'Cannot send message, not connected.'))
            return False
        try:
            with self.lock:
                if self.ws:
                    self.ws.send(json.dumps(message_data))
                    return True
            return False
        except Exception as e:
            self.message_queue.put(('error', f'Failed to send message: {str(e)}'))
            return False

# Global WebSocket manager
if 'ws_manager' not in st.session_state:
    st.session_state.ws_manager = WebSocketManager()

def load_conversations():
    """Load list of conversations"""
    try:
        response = requests.get(f"{BASE_URL}/api/conversations", timeout=5)
        response.raise_for_status()
        st.session_state.conversations = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading conversations: {str(e)}")
        return False
    return True

def create_new_conversation():
    """Create a new conversation"""
    try:
        response = requests.post(f"{BASE_URL}/api/conversations", timeout=5)
        response.raise_for_status()
        data = response.json()
        st.session_state.conversation_id = data.get('conversation_id') or data.get('id')
        st.session_state.messages = []
        st.session_state.message_count = 0
        if st.session_state.conversation_id:
            st.session_state.ws_manager.connect(st.session_state.conversation_id)
        load_conversations()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating conversation: {str(e)}")
        return False

def delete_conversation(conv_id):
    """Delete a conversation"""
    try:
        response = requests.delete(f"{BASE_URL}/api/conversations/{conv_id}", timeout=5)
        response.raise_for_status()
        if conv_id == st.session_state.conversation_id:
            create_new_conversation()
        else:
            load_conversations()
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting conversation: {str(e)}")

def switch_conversation(conv_id):
    """Switch to a different conversation"""
    if conv_id == st.session_state.conversation_id:
        return  # Already on this conversation
        
    # Disconnect from current conversation
    st.session_state.ws_manager.disconnect()
    
    # Clear current state
    st.session_state.conversation_id = conv_id
    st.session_state.messages = []
    st.session_state.message_count = 0
    st.session_state.is_typing = False
    
    # Load conversation history first
    load_conversation_history(conv_id)
    
    # Then connect to WebSocket
    st.session_state.ws_manager.connect(conv_id)

def load_conversation_history(conv_id):
    """Load conversation history from the API"""
    try:
        response = requests.get(f"{BASE_URL}/api/conversations/{conv_id}", timeout=5)
        response.raise_for_status()
        messages = response.json()
        st.session_state.messages = messages if isinstance(messages, list) else []
        st.session_state.message_count = len(st.session_state.messages)
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading conversation history: {str(e)}")

def process_websocket_messages():
    """Process incoming WebSocket messages"""
    ws_manager = st.session_state.ws_manager
    
    # Update connection status
    st.session_state.ws_connected = ws_manager.is_connected()
    
    # Process messages
    messages_processed = 0
    while not ws_manager.message_queue.empty() and messages_processed < 5:
        try:
            msg_type, data = ws_manager.message_queue.get_nowait()
            messages_processed += 1
            
            if msg_type == 'message':
                if data.get('type') == 'history':
                    # Only update history if we don't already have messages
                    # This prevents old conversations from showing in new ones
                    if not st.session_state.messages:
                        history = data.get('history', [])
                        if isinstance(history, list):
                            st.session_state.messages = history
                            st.session_state.message_count = len(st.session_state.messages)
                    st.session_state.is_typing = False
                    
                elif data.get('type') == 'response':
                    content = data.get('content', '')
                    if content:
                        assistant_message = {
                            'id': str(datetime.now().timestamp()),
                            'role': 'assistant',
                            'content': content,
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.messages.append(assistant_message)
                        st.session_state.message_count = len(st.session_state.messages)
                    st.session_state.is_typing = False
                    
                elif data.get('type') == 'typing':
                    st.session_state.is_typing = data.get('state') == 'processing'
                    
                elif data.get('type') == 'error':
                    st.error(f"Server error: {data.get('message', 'Unknown error')}")
                    st.session_state.is_typing = False
                    
            elif msg_type == 'error':
                # Don't show connection errors as they're handled elsewhere
                if "Connection failed" not in data and "Connection closed" not in data:
                    st.error(f"WebSocket error: {data}")
                st.session_state.is_typing = False
                
            elif msg_type == 'status':
                # Only show status updates briefly
                pass  # Remove automatic status display
                
        except queue.Empty:
            break
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
    
    return messages_processed > 0

def reconnect_websocket():
    """Reconnect WebSocket"""
    if st.session_state.conversation_id:
        st.session_state.ws_manager.disconnect()
        time.sleep(1)  # Brief pause
        st.session_state.ws_manager.connect(st.session_state.conversation_id)

# Initialize application
if not st.session_state.conversation_id:
    with st.spinner("Creating initial conversation..."):
        if not create_new_conversation():
            st.error("Failed to create initial conversation. Please check if the server is running.")
            st.stop()
        # Give time for WebSocket to connect
        time.sleep(1)

# Process WebSocket messages first
message_updates = process_websocket_messages()

# Sidebar for conversations
with st.sidebar:
    st.title("💬 Conversations")
    
    if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
        with st.spinner("Creating new conversation..."):
            if create_new_conversation():
                st.success("New conversation created!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Failed to create new conversation")
    
    st.subheader("Recent Chats")
    
    # Load conversations if empty
    if not st.session_state.conversations:
        with st.spinner("Loading conversations..."):
            load_conversations()
    
    if st.session_state.conversations:
        for conv in st.session_state.conversations:
            col1, col2 = st.columns([4, 1])
            with col1:
                is_current = conv.get('id') == st.session_state.conversation_id
                button_style = "🔵" if is_current else "💬"
                title = conv.get('title', conv.get('name', 'New Chat'))[:20]
                if len(conv.get('title', conv.get('name', ''))) > 20:
                    title += "..."
                
                if st.button(
                    f"{button_style} {title}",
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    disabled=is_current
                ):
                    switch_conversation(conv['id'])
                    time.sleep(0.5)  # Brief pause for state to update
                    st.rerun()
                    
            with col2:
                if st.button("🗑️", key=f"del_{conv['id']}", help="Delete conversation"):
                    if st.session_state.get(f"confirm_delete_{conv['id']}", False):
                        delete_conversation(conv['id'])
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{conv['id']}"] = True
                        st.rerun()
    else:
        st.info("No conversations found")

# Main chat interface
st.title("🤖 Agentic Chat")

# Connection status and controls
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.session_state.ws_connected:
        st.success("🟢 Connected")
    else:
        st.error("🔴 Disconnected")

with col2:
    if st.button("🔄 Reconnect", key="reconnect_btn"):
        with st.spinner("Reconnecting..."):
            reconnect_websocket()
            time.sleep(1)
            st.rerun()

with col3:
    st.info(f"Messages: {st.session_state.message_count}")

# Display messages in a container
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if st.session_state.is_typing:
        with st.chat_message("assistant"):
            st.markdown("🤔 Thinking...")

# Message input
user_input = st.chat_input(
    "Type your message here...",
    disabled=not st.session_state.ws_connected,
    key="chat_input"
)

if user_input and not st.session_state.processing_message:
    # Prevent multiple simultaneous message processing
    st.session_state.processing_message = True
    
    # Check rate limiting
    current_time = time.time()
    if current_time - st.session_state.last_message_time < 1:  # 1 second cooldown
        st.warning("Please wait before sending another message")
        st.session_state.processing_message = False
    else:
        st.session_state.last_message_time = current_time
        
        # Add user message
        user_message = {
            'id': str(datetime.now().timestamp()),
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.messages.append(user_message)
        st.session_state.message_count = len(st.session_state.messages)

        # Handle special case
        if user_input.lower() == "who is rajeev":
            assistant_message = {
                'id': str(datetime.now().timestamp()),
                'role': 'assistant',
                'content': "Rajeev is a talented and dedicated software engineer at Opstree. He is known for his expertise in building scalable and robust applications. He is passionate about AI and is always exploring new technologies to solve complex problems. In his free time, he enjoys contributing to open-source projects and mentoring junior developers.",
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            st.session_state.message_count = len(st.session_state.messages)
            st.session_state.is_typing = False
            st.session_state.processing_message = False
        else:
            # Send message via WebSocket
            message_data = {
                'text': user_input,
                'type': 'message',
                'content': user_input + ". explain or tell me about this"
            }
            
            if st.session_state.ws_manager.send_message(message_data):
                st.session_state.is_typing = True
            else:
                st.error("Failed to send message - WebSocket not connected")
                st.session_state.is_typing = False
            st.session_state.processing_message = False
        
        st.rerun()

# Auto-refresh logic - only refresh when necessary
if message_updates or st.session_state.is_typing:
    # Only rerun if we processed messages or are typing
    time.sleep(0.5)
    st.rerun()
elif not st.session_state.ws_manager.message_queue.empty():
    # Process any remaining messages
    st.rerun()