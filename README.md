# Agentic Chat with Streamlit

A Streamlit-based chat interface that connects to an MCP (Model Control Protocol) server, providing a user-friendly way to interact with LangGraph agents and various tools.

## Features

- Real-time chat interface with WebSocket support
- Conversation history and management
- Simple and intuitive UI built with Streamlit
- Support for multiple conversations
- Integration with MCP server for agent-based interactions

## Prerequisites

- Python 3.8+
- MCP Server (local or remote) running on `http://localhost:8000`

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic_chat
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the MCP Server**
   Make sure your MCP server is running on `http://localhost:8000`

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Access the application**
   Open your web browser and navigate to `http://localhost:8501`

## Usage

- **New Chat**: Click the "New Chat" button in the sidebar to start a new conversation
- **Switch Conversations**: Click on any conversation in the sidebar to switch to it
- **Delete Conversations**: Click the 🗑️ icon next to a conversation to delete it
- **Send Messages**: Type your message in the input box at the bottom and press Enter to send

## Configuration

Edit the `.env` file to configure the following settings:

```env
# Google API Key for Gemini model
GOOGLE_API_KEY=your_google_api_key

# MCP Server Configuration
MCP_SERVER_URL=http://localhost:8000/sse
MCP_SERVER_TRANSPORT=sse

# Chat Server Configuration
CHAT_SERVER_HOST=0.0.0.0
CHAT_SERVER_PORT=8001

# Logging Level
LOG_LEVEL=INFO
```

## Running the Application

1. **Start the MCP Server** (if not already running)
   ```bash
   # Example - adjust based on your MCP server setup
   python mcp_server.py
   ```

2. **Start the Chat Server**
   ```bash
   uvicorn chat_server:app --reload --host 0.0.0.0 --port 8001
   ```

3. **Access the Web Interface**
   Open your browser and navigate to:
   ```
   http://localhost:8001
   ```

## Project Structure

- `chat_server.py`: FastAPI server with WebSocket endpoints and LangGraph agent integration
- `chat_client.py`: React-based frontend for the chat interface
- `setup.py`: Setup script for environment configuration
- `requirements.txt`: Python dependencies
- `conversations/`: Directory for storing conversation history
- `static/`: Frontend static files
- `logs/`: Application logs

## API Endpoints

- `GET /`: Web interface
- `WS /ws/{conversation_id}`: WebSocket endpoint for real-time chat
- `GET /api/conversations`: List all conversations
- `POST /api/conversations`: Create a new conversation
- `GET /api/conversations/{conversation_id}`: Get conversation history
- `DELETE /api/conversations/{conversation_id}`: Delete a conversation
- `GET /api/tools`: List available MCP tools
- `GET /health`: Health check endpoint

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Google API Key for Gemini model | Yes | - |
| `MCP_SERVER_URL` | URL of the MCP server | Yes | - |
| `MCP_SERVER_TRANSPORT` | Transport protocol for MCP | No | `sse` |
| `CHAT_SERVER_HOST` | Host to run the chat server on | No | `0.0.0.0` |
| `CHAT_SERVER_PORT` | Port to run the chat server on | No | `8001` |
| `LOG_LEVEL` | Logging level | No | `INFO` |

## Development

### Frontend Development

The frontend is built with React. To develop the frontend:

1. Navigate to the static directory
   ```bash
   cd static
   ```

2. Install dependencies
   ```bash
   npm install
   ```

3. Start the development server
   ```bash
   npm start
   ```

### Backend Development

The backend is built with FastAPI. To run the development server:

```bash
uvicorn chat_server:app --reload --host 0.0.0.0 --port 8001
```

## Troubleshooting

1. **MCP Server Connection Issues**
   - Ensure the MCP server is running and accessible
   - Check the MCP server URL in the `.env` file
   - Verify network connectivity between the chat server and MCP server

2. **Missing Dependencies**
   - Run `pip install -r requirements.txt` to install all required packages
   - Ensure you're using the correct Python version (3.8+)

3. **Google API Key Issues**
   - Verify your Google API key is valid and has access to the Gemini API
   - Check the `.env` file for the correct API key

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

