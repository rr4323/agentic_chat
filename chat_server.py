"""
FastAPI WebSocket Chat Server with LangGraph MCP Agent Integration

This server provides a WebSocket-based chat interface that integrates with
MCP servers using LangGraph agents for advanced tool usage and reasoning.
"""

import os
import sys
import logging
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import aiofiles
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for required packages and install if needed
required_packages = ["langchain-mcp-adapters", "langchain-google-genai", "langgraph"]
for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        logger.info(f"Installing {package}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Pydantic models
class Message(BaseModel):
    id: str
    role: str  # 'user' or 'assistant' or 'system'
    content: str
    timestamp: str
    conversation_id: str

class ChatMessage(BaseModel):
    type: str
    content: Optional[str] = None
    conversation_id: Optional[str] = None

class ConversationInfo(BaseModel):
    id: str
    name: str
    last_message: Optional[str] = None
    created_at: str
    message_count: int

class MCPServerConfig(BaseModel):
    name: str
    url: str
    transport: str = "sse"

# LangGraph MCP Agent
class LangGraphMCPAgent:
    def __init__(self, mcp_servers: List[MCPServerConfig]):
        self.mcp_servers = mcp_servers
        self.conversation_history: List[Message] = []
        self.state = 'user'  # 'user' | 'processing' | 'tool_use'
        self.client = None
        self.agent = None
        self.tools = []
        
    async def initialize(self):
        """Initialize the MCP client and LangGraph agent"""
        try:
            # Get Google API key
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            # Create MCP client configuration
            server_config = {}
            for server in self.mcp_servers:
                server_config[server.name] = {
                    "url": server.url,
                    "transport": server.transport
                }
            
            # Initialize MCP client
            logger.info("Initializing MCP client...")
            self.client = MultiServerMCPClient(server_config)
            
            # Get tools from MCP servers
            logger.info("Fetching tools from MCP servers...")
            self.tools = await self.client.get_tools()
            logger.info(f"Available tools: {[tool.name for tool in self.tools]}")
            
            # Create LangGraph agent
            logger.info("Creating LangGraph agent...")
            model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.7,
                convert_system_message_to_human=True
            )
            
            self.agent = create_react_agent(
                model=model,
                tools=self.tools,
                prompt="""
You are a helpful AI assistant with access to various tools for analysis and knowledge retrieval.
When a user's query requires analysis, data processing, or external knowledge, use the appropriate tools.
Always provide clear, informative responses and explain your reasoning when using tools.
Be conversational and helpful while maintaining accuracy.
"""
            )
            
            logger.info("LangGraph MCP Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            raise
    
    def list_tools(self) -> List[str]:
        """List available tools"""
        return [tool.name for tool in self.tools]
    
    async def process_message(self, message: str, conversation_id: str) -> str:
        """Process user message using LangGraph agent"""
        self.state = 'processing'
        
        # Add user message to history
        user_message = Message(
            id=str(uuid.uuid4()),
            role='user',
            content=message,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id
        )
        self.conversation_history.append(user_message)
        
        try:
            # Prepare conversation context
            messages = []
            
            # Add recent conversation history for context (last 10 messages)
            recent_history = self.get_history(conversation_id)[-10:]
            for msg in recent_history[:-1]:  # Exclude the current message
                if msg.role == 'user':
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == 'assistant':
                    messages.append(AIMessage(content=msg.content))
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            # Check if tools might be needed
            if any(keyword in message.lower() for keyword in ['analyze', 'extract', 'sentiment', 'fact', 'check', 'company', 'entity']):
                self.state = 'tool_use'
            
            # Process with LangGraph agent
            logger.info(f"Processing message with agent: {message}")
            response = await self.agent.ainvoke({"messages": messages})
            
            # Extract response content
            response_content = ""
            if isinstance(response, dict) and "messages" in response:
                messages_response = response["messages"]
                for msg in messages_response:
                    
                    if isinstance(msg, AIMessage):
                        response_content = msg.content
            else:
                response_content = str(response)
            # Add agent response to history
            agent_message = Message(
                id=str(uuid.uuid4()),
                role='assistant',
                content=response_content,
                timestamp=datetime.now().isoformat(),
                conversation_id=conversation_id
            )
            self.conversation_history.append(agent_message)
            
            self.state = 'user'
            return response_content
            
        except Exception as e:
            self.state = 'user'
            logger.error(f"Error processing message: {e}")
            error_message = f"I apologize, but I encountered an error while processing your message: {str(e)}"
            
            # Add error message to history
            error_msg = Message(
                id=str(uuid.uuid4()),
                role='assistant',
                content=error_message,
                timestamp=datetime.now().isoformat(),
                conversation_id=conversation_id
            )
            self.conversation_history.append(error_msg)
            
            return error_message
    
    def get_history(self, conversation_id: str) -> List[Message]:
        """Get conversation history for a specific conversation"""
        return [msg for msg in self.conversation_history if msg.conversation_id == conversation_id]
    
    async def load_history(self, conversation_id: str) -> List[Message]:
        """Load conversation history from storage"""
        try:
            file_path = Path(f"conversations/{conversation_id}.json")
            if file_path.exists():
                async with aiofiles.open(file_path, 'r') as f:
                    data = await f.read()
                    history_data = json.loads(data)
                    history = [Message(**msg) for msg in history_data]
                    self.conversation_history.extend(history)
                    return history
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}")
        return []
    
    async def save_history(self, conversation_id: str):
        """Save conversation history to storage"""
        try:
            os.makedirs("conversations", exist_ok=True)
            file_path = Path(f"conversations/{conversation_id}.json")
            
            history = self.get_history(conversation_id)
            history_data = [msg.model_dump() for msg in history]
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(history_data, indent=2))
                
        except Exception as e:
            logger.error(f"Error saving conversation {conversation_id}: {e}")

# Conversation manager
class ConversationManager:
    def __init__(self, mcp_servers: List[MCPServerConfig]):
        self.mcp_servers = mcp_servers
        self.conversations: Dict[str, LangGraphMCPAgent] = {}
    
    async def get_or_create_conversation(self, conversation_id: str) -> LangGraphMCPAgent:
        """Get existing conversation or create new one"""
        if conversation_id not in self.conversations:
            agent = LangGraphMCPAgent(self.mcp_servers)
            await agent.initialize()
            self.conversations[conversation_id] = agent
        
        return self.conversations[conversation_id]
    
    async def load_conversation(self, conversation_id: str) -> LangGraphMCPAgent:
        """Load conversation with history"""
        agent = await self.get_or_create_conversation(conversation_id)
        await agent.load_history(conversation_id)
        return agent

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, conversation_id: str):
        await websocket.accept()
        self.active_connections[conversation_id] = websocket
    
    def disconnect(self, conversation_id: str):
        if conversation_id in self.active_connections:
            del self.active_connections[conversation_id]
    
    async def send_message(self, conversation_id: str, message: dict):
        if conversation_id in self.active_connections:
            await self.active_connections[conversation_id].send_text(json.dumps(message))

# FastAPI app setup
app = FastAPI(title="LangGraph MCP Agent Chat API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP Server Configuration
MCP_SERVERS = [
    MCPServerConfig(
        name="knowledge_tools",
        url="http://localhost:8000/sse",
        transport="sse"
    )
]

# Global managers
conversation_manager = ConversationManager(MCP_SERVERS)
connection_manager = ConnectionManager()

# Serve static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket endpoint
@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    await connection_manager.connect(websocket, conversation_id)
    
    try:
        # Load conversation
        agent = await conversation_manager.load_conversation(conversation_id)
        
        # Send initial data
        history = agent.get_history(conversation_id)
        await websocket.send_text(json.dumps({
            "type": "history",
            "conversation_id": conversation_id,
            "history": [msg.dict() for msg in history],
            "state": agent.state
        }))
        
        # Send init complete with tools info
        await websocket.send_text(json.dumps({
            "type": "init_complete",
            "conversation_id": conversation_id,
            "state": agent.state,
            "tools": agent.list_tools()
        }))
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            print(data)
            message_data = json.loads(data)
            
            if message_data["type"] == "message":
                # Send typing indicator
                await websocket.send_text(json.dumps({
                    "type": "typing",
                    "state": "processing"
                }))
                
                try:
                    # Process message with LangGraph agent
                    response = await agent.process_message(
                        message_data["content"], 
                        conversation_id
                    )
                    
                    # Save conversation
                    await agent.save_history(conversation_id)
                    
                    # Send response
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "content": response,
                        "state": agent.state,
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Error processing message: {str(e)}"
                    }))
            
            elif message_data["type"] == "get_tools":
                await websocket.send_text(json.dumps({
                    "type": "tools",
                    "tools": agent.list_tools()
                }))
            
            elif message_data["type"] == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    
    except WebSocketDisconnect:
        connection_manager.disconnect(conversation_id)
        if conversation_id in conversation_manager.conversations:
            agent = conversation_manager.conversations[conversation_id]
            await agent.save_history(conversation_id)
        logger.info(f"Client {conversation_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(conversation_id)

# REST API endpoints
@app.get("/api/conversations", response_model=List[ConversationInfo])
async def get_conversations():
    """Get list of all conversations"""
    conversations = []
    conversations_dir = Path("conversations")
    
    if conversations_dir.exists():
        for file_path in conversations_dir.glob("*.json"):
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    data = await f.read()
                    history_data = json.loads(data)
                
                if history_data:
                    conversation_id = file_path.stem
                    last_message = history_data[-1]["content"] if history_data else None
                    created_at = history_data[0]["timestamp"] if history_data else datetime.now().isoformat()
                    
                    conversations.append(ConversationInfo(
                        id=conversation_id,
                        name=f"Chat {conversation_id[:8]}",
                        last_message=last_message,
                        created_at=created_at,
                        message_count=len(history_data)
                    ))
            except Exception as e:
                logger.error(f"Error reading conversation {file_path}: {e}")
    
    return conversations

@app.get("/api/conversations/{conversation_id}", response_model=List[Message])
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    try:
        file_path = Path(f"conversations/{conversation_id}.json")
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
                data = await f.read()
                history_data = json.loads(data)
                return [Message(**msg) for msg in history_data]
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        file_path = Path(f"conversations/{conversation_id}.json")
        if file_path.exists():
            file_path.unlink()
        
        # Remove from active conversations
        if conversation_id in conversation_manager.conversations:
            del conversation_manager.conversations[conversation_id]
        
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations")
async def create_conversation():
    """Create a new conversation"""
    conversation_id = str(uuid.uuid4())
    return {"conversation_id": conversation_id}

@app.get("/api/tools")
async def get_available_tools():
    """Get list of available MCP tools"""
    try:
        # Create a temporary agent to get tools
        temp_agent = LangGraphMCPAgent(MCP_SERVERS)
        await temp_agent.initialize()
        tools = temp_agent.list_tools()
        return {"tools": tools}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Configuration endpoint
@app.get("/api/config")
async def get_config():
    """Get server configuration"""
    return {
        "mcp_servers": [server.dict() for server in MCP_SERVERS],
        "google_api_key_configured": bool(os.getenv("GOOGLE_API_KEY"))
    }

# Main frontend route
@app.get("/")
async def get_frontend():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph MCP Agent Chat</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
        <div id="app">
            <h1>LangGraph MCP Agent Chat</h1>
            <p>Advanced AI agent with MCP tool integration</p>
            <p>WebSocket endpoint: ws://localhost:8000/ws/{conversation_id}</p>
            <p>Make sure your MCP server is running on localhost:8000/sse</p>
        </div>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
