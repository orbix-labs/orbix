import os
import json
import hmac
import hashlib
import time
import asyncio
import threading
import logging
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

class ThreadManager:
    """Manages LangGraph thread states for Slack conversations"""
    
    def __init__(self, max_threads: int = 1000, ttl_hours: int = 24):
        self.max_threads = max_threads
        self.ttl_hours = ttl_hours
        self._lock = threading.Lock()
        # Track thread metadata
        self.thread_metadata: Dict[str, Dict[str, Any]] = {}
        
    def get_thread_config(self, thread_id: str, channel_id: str) -> Dict[str, Any]:
        """Get thread configuration for LangGraph"""
        with self._lock:
            # Clean up expired threads first
            self._cleanup_expired_threads()
            
            if thread_id not in self.thread_metadata:
                # Create new thread metadata
                self.thread_metadata[thread_id] = {
                    "channel_id": channel_id,
                    "created_at": datetime.now(),
                    "last_activity": datetime.now(),
                    "message_count": 0
                }
                
                # Cleanup if we have too many threads
                if len(self.thread_metadata) > self.max_threads:
                    self._cleanup_oldest_threads()
            else:
                # Update last activity
                self.thread_metadata[thread_id]["last_activity"] = datetime.now()
            
            return {"configurable": {"thread_id": thread_id}}
    
    def increment_message_count(self, thread_id: str):
        """Increment message count for a thread"""
        with self._lock:
            if thread_id in self.thread_metadata:
                self.thread_metadata[thread_id]["message_count"] += 1
                self.thread_metadata[thread_id]["last_activity"] = datetime.now()
    
    def _cleanup_expired_threads(self):
        """Remove expired thread metadata"""
        current_time = datetime.now()
        expired_threads = [
            thread_id for thread_id, metadata in self.thread_metadata.items()
            if (current_time - metadata["last_activity"]).total_seconds() > (self.ttl_hours * 3600)
        ]
        
        for thread_id in expired_threads:
            del self.thread_metadata[thread_id]
        
        if expired_threads:
            logger.info(f"Cleaned up {len(expired_threads)} expired thread metadata")
    
    def _cleanup_oldest_threads(self):
        """Remove oldest thread metadata when limit is reached"""
        if len(self.thread_metadata) <= self.max_threads:
            return
        
        # Sort by last activity and remove oldest
        sorted_threads = sorted(
            self.thread_metadata.items(),
            key=lambda x: x[1]["last_activity"]
        )
        
        threads_to_remove = len(self.thread_metadata) - self.max_threads + 1
        for thread_id, _ in sorted_threads[:threads_to_remove]:
            del self.thread_metadata[thread_id]
        
        logger.info(f"Cleaned up {threads_to_remove} oldest thread metadata")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about thread states"""
        with self._lock:
            return {
                "total_threads": len(self.thread_metadata),
                "max_threads": self.max_threads,
                "ttl_hours": self.ttl_hours,
                "threads": [
                    {
                        "thread_id": thread_id,
                        "channel_id": metadata["channel_id"],
                        "message_count": metadata["message_count"],
                        "last_activity": metadata["last_activity"].isoformat(),
                        "age_hours": (datetime.now() - metadata["created_at"]).total_seconds() / 3600
                    }
                    for thread_id, metadata in self.thread_metadata.items()
                ]
            }
    
    def delete_thread(self, thread_id: str) -> bool:
        """Delete a specific thread"""
        with self._lock:
            if thread_id in self.thread_metadata:
                del self.thread_metadata[thread_id]
                return True
            return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize thread manager and memory saver
thread_manager = ThreadManager(max_threads=1000, ttl_hours=24)
memory_saver = MemorySaver()

def debug_environment():
    """Debug function to check environment variables and system info"""
    logger.info("=== ENVIRONMENT DEBUG ===")
    
    # Check important environment variables
    env_vars = [
        'SLACK_BOT_TOKEN', 'SLACK_SIGNING_SECRET', 
        'MCP_SERVER_COMMAND', 'MCP_SERVER_ARGS',
        'KUBECONFIG', 'KUBERNETES_SERVICE_HOST', 'KUBERNETES_SERVICE_PORT',
        'OPENAI_API_KEY', 'HOME', 'USER', 'PWD'
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if var in ['SLACK_BOT_TOKEN', 'SLACK_SIGNING_SECRET', 'OPENAI_API_KEY']:
            logger.info(f"{var}: {'SET (' + str(len(value)) + ' chars)' if value else 'NOT SET'}")
        else:
            logger.info(f"{var}: {value}")
    
    # Check if kubectl is available
    try:
        import subprocess
        result = subprocess.run(['kubectl', 'version', '--client'], 
                              capture_output=True, text=True, timeout=10)
        logger.info(f"kubectl client version: {result.stdout.strip()}")
    except Exception as e:
        logger.warning(f"kubectl not available: {e}")
    
    # Check if kubeconfig file exists
    kubeconfig_path = os.path.expanduser('~/.kube/config')
    if os.path.exists(kubeconfig_path):
        logger.info(f"kubeconfig file exists at: {kubeconfig_path}")
        try:
            with open(kubeconfig_path, 'r') as f:
                content = f.read()
                logger.info(f"kubeconfig file size: {len(content)} bytes")
        except Exception as e:
            logger.warning(f"Could not read kubeconfig: {e}")
    else:
        logger.info("No kubeconfig file found at ~/.kube/config")
    
    # Check MCP server path
    mcp_server_path = "/app/mcp-server-kubernetes/dist/index.js"
    if os.path.exists(mcp_server_path):
        logger.info(f"MCP server file exists: {mcp_server_path}")
    else:
        logger.warning(f"MCP server file not found: {mcp_server_path}")
        # Try to find alternative paths
        alt_paths = [
            "/app/mcp-server-kubernetes/src/index.ts",
            "./mcp-server-kubernetes/dist/index.js",
            "./mcp-server-kubernetes/src/index.ts"
        ]
        for path in alt_paths:
            if os.path.exists(path):
                logger.info(f"Alternative MCP server file found: {path}")
                break
    
    logger.info("=== END ENVIRONMENT DEBUG ===")

class SlackBot:
    def __init__(self):
        # Prepare environment variables for MCP server
        mcp_env = os.environ.copy()  # Copy all current environment variables
        
        # Ensure Kubernetes environment variables are available
        if not mcp_env.get('KUBERNETES_SERVICE_HOST'):
            logger.warning("KUBERNETES_SERVICE_HOST not found in environment")
        if not mcp_env.get('KUBERNETES_SERVICE_PORT'):
            logger.warning("KUBERNETES_SERVICE_PORT not found in environment")
            
        self.server_params = StdioServerParameters(
            command=os.environ.get('MCP_SERVER_COMMAND', "node"),
            args=[os.environ.get('MCP_SERVER_ARGS', "/app/mcp-server-kubernetes/dist/index.js")],
            env=mcp_env  # Pass environment variables to child process
        )
        logger.info(f"SlackBot initialized with MCP server command: {self.server_params.command}")
        logger.info(f"MCP server args: {self.server_params.args}")
        
        # Log environment variables for debugging
        logger.info("Environment variables check:")
        for key in ['SLACK_BOT_TOKEN', 'SLACK_SIGNING_SECRET', 'MCP_SERVER_COMMAND', 'MCP_SERVER_ARGS', 'KUBERNETES_SERVICE_HOST', 'KUBERNETES_SERVICE_PORT']:
            value = os.environ.get(key)
            if key in ['SLACK_BOT_TOKEN', 'SLACK_SIGNING_SECRET']:
                logger.info(f"{key}: {'SET' if value else 'NOT SET'}")
            else:
                logger.info(f"{key}: {value}")
    
    async def test_mcp_connection(self):
        """Test MCP server connection before processing questions"""
        try:
            logger.info("Testing MCP server connection...")
            async with stdio_client(self.server_params) as (read, write):
                logger.info("MCP client connection successful")
                async with ClientSession(read, write) as session:
                    logger.info("Initializing MCP session for test...")
                    await session.initialize()
                    logger.info("MCP session test successful")
                    return True
        except Exception as e:
            logger.error(f"MCP connection test failed: {e}")
            import traceback
            logger.error(f"MCP test traceback: {traceback.format_exc()}")
            return False

    async def process_question_with_thread(self, question, thread_config):
        """Process a question using LangGraph with thread state management"""
        try:
            logger.info(f"Processing question: {question}")
            logger.info(f"Thread config: {thread_config}")
            
            async with stdio_client(self.server_params) as (read, write):
                logger.info("MCP client connected successfully")
                async with ClientSession(read, write) as session:
                    # Initialize the connection
                    logger.info("Initializing MCP session...")
                    await session.initialize()
                    logger.info("MCP session initialized successfully")
                    
                    # Get tools
                    logger.info("Loading MCP tools...")
                    tools = await load_mcp_tools(session)
                    logger.info(f"Successfully loaded {len(tools)} MCP tools")
                    
                    # Log available tools
                    for i, tool in enumerate(tools):
                        logger.info(f"Tool {i+1}: {tool.name if hasattr(tool, 'name') else 'Unknown'}")
                    
                    # Create agent with memory saver for thread state management
                    logger.info("Creating ReAct agent with memory...")
                    agent = create_react_agent("openai:gpt-4o-mini", tools, checkpointer=memory_saver)
                    
                    # Invoke agent with thread configuration
                    logger.info("Invoking agent with thread context...")
                    agent_response = await agent.ainvoke(
                        {"messages": [HumanMessage(content=question)]}, 
                        config=thread_config
                    )
                    
                    # Extract the final answer
                    final_message = agent_response["messages"][-1]
                    logger.info(f"Agent response received: {final_message.content[:200]}...")
                    return final_message.content
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error processing question: {str(e)}"

# Run environment debug and initialize bot
debug_environment()
bot = SlackBot()

def verify_slack_request(request):
    """Verify that the request is from Slack"""
    signing_secret = os.environ.get('SLACK_SIGNING_SECRET')
    if not signing_secret:
        logger.warning("SLACK_SIGNING_SECRET not set - skipping signature verification")
        return False
    
    timestamp = request.headers.get('X-Slack-Request-Timestamp')
    slack_signature = request.headers.get('X-Slack-Signature')
    
    if not timestamp or not slack_signature:
        logger.warning("Missing timestamp or signature in request headers")
        return False
    
    # Verify timestamp is within 5 minutes
    if abs(time.time() - int(timestamp)) > 300:
        logger.warning("Request timestamp is too old")
        return False
    
    # Create signature
    request_body = request.get_data()
    basestring = f"v0:{timestamp}:{request_body.decode()}"
    
    my_signature = 'v0=' + hmac.new(
        signing_secret.encode(),
        basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    is_valid = hmac.compare_digest(my_signature, slack_signature)
    logger.info(f"Slack signature verification: {'PASSED' if is_valid else 'FAILED'}")
    return is_valid

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker"""
    logger.info("Health check requested")
    return jsonify({'status': 'healthy', 'service': 'kubernetes-ai-slack-bot'}), 200

@app.route('/test-mcp', methods=['GET'])
def test_mcp_endpoint():
    """Test MCP server connection - REQUIRE AUTHENTICATION IN PRODUCTION"""
    logger.info("MCP test endpoint requested")
    
    # TODO: Add authentication/authorization check here
    # if not is_authorized(request):
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    def run_async_test():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(bot.test_mcp_connection())
            return result
        except Exception as e:
            logger.error(f"Async MCP test error: {e}")
            return False
        finally:
            loop.close()
    
    try:
        success = run_async_test()
        return jsonify({
            'status': 'success' if success else 'failed',
            'message': 'MCP connection test completed',
            'connection_successful': success
        }), 200 if success else 500
    except Exception as e:
        logger.error(f"MCP test endpoint error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/threads', methods=['GET'])
def get_thread_stats():
    """Get thread statistics - REQUIRE AUTHENTICATION IN PRODUCTION"""
    logger.info("Thread statistics endpoint requested")
    
    # TODO: Add authentication/authorization check here
    # if not is_authorized(request):
    #     return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        stats = thread_manager.get_stats()
        # Remove sensitive thread details
        sanitized_stats = {
            'total_threads': stats['total_threads'],
            'max_threads': stats['max_threads'],
            'ttl_hours': stats['ttl_hours']
            # Removed individual thread details for security
        }
        return jsonify({
            'status': 'success',
            'data': sanitized_stats
        }), 200
    except Exception as e:
        logger.error(f"Thread stats endpoint error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/slack/events', methods=['POST'])
def handle_slack_event():
    logger.info(f"Received Slack event: {request.method} {request.path}")
    logger.info(f"Content-Type: {request.content_type}")
    
    # Verify the request is from Slack
    if not verify_slack_request(request):
        logger.warning("Slack signature verification failed")
        return jsonify({'error': 'Invalid signature'}), 401
    
    # Handle URL verification
    if request.json and request.json.get('type') == 'url_verification':
        challenge = request.json.get('challenge')
        logger.info(f"URL verification challenge received")
        return jsonify({'challenge': challenge})
    
    # Handle slash commands
    if request.content_type == 'application/x-www-form-urlencoded':
        logger.info("Processing slash command")
        return handle_slash_command()
    
    # Handle app mentions and messages
    event_data = request.json
    logger.info("Processing Slack event data")
    
    if event_data and 'event' in event_data:
        event = event_data['event']
        
        # Ignore bot messages to prevent loops
        if event.get('bot_id'):
            logger.info("Ignoring bot message to prevent loops")
            return '', 200
        
        # Only handle messages and app mentions
        if event.get('type') in ['message', 'app_mention']:
            logger.info(f"Processing {event.get('type')} event")
            # Process in background to avoid timeout
            def run_async():
                try:
                    logger.info("Starting async thread for message processing")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(process_slack_message(event))
                    logger.info("Async message processing completed")
                except Exception as e:
                    logger.error(f"Error in async thread: {e}")
                    import traceback
                    logger.error(f"Async thread traceback: {traceback.format_exc()}")
                finally:
                    loop.close()
                    logger.info("Async thread loop closed")
            
            thread = threading.Thread(target=run_async)
            thread.daemon = True  # Make thread daemon to avoid hanging
            thread.start()
            logger.info("Background thread started for message processing")
    
    return '', 200

def handle_slash_command():
    """Handle Slack slash commands"""
    command_text = request.form.get('text', '').lower()
    channel_id = request.form.get('channel_id')
    user_id = request.form.get('user_id')
    
    logger.info(f"Slash command received: '{command_text}' from user {user_id} in channel {channel_id}")
    
    # Process the command
    if 'help' in command_text:
        response = {
            'response_type': 'in_channel',
            'text': 'Available commands:\n• Ask any Kubernetes question\n• Use /k8s <your question>\n• help - Show this help message'
        }
        logger.info("Returning help response")
    else:
        # For slash commands, we'll return a quick response and process async
        response = {
            'response_type': 'in_channel',
            'text': 'Processing your Kubernetes question...'
        }
        
        # Process the actual question in background
        def run_async():
            try:
                logger.info("Starting async thread for slash command processing")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(process_slash_command_async(command_text, channel_id, user_id))
                logger.info("Async slash command processing completed")
            except Exception as e:
                logger.error(f"Error in slash command async thread: {e}")
                import traceback
                logger.error(f"Slash command async thread traceback: {traceback.format_exc()}")
            finally:
                loop.close()
                logger.info("Slash command async thread loop closed")
        
        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()
        logger.info("Background thread started for slash command processing")
    
    return jsonify(response)

async def process_slash_command_async(question, channel_id, user_id):
    """Process slash command asynchronously using LangGraph thread management"""
    logger.info(f"Processing slash command async: '{question}' for user {user_id}")
    
    if not question or question == 'help':
        logger.info("Skipping empty or help command")
        return
    
    # Use unique thread_id for slash commands (they create new conversations)
    thread_id = f"slash_{channel_id}_{user_id}_{int(time.time())}"
    
    try:
        # Get thread configuration for LangGraph
        thread_config = thread_manager.get_thread_config(thread_id, channel_id)
        
        logger.info("Processing slash command with LangGraph thread management...")
        answer = await bot.process_question_with_thread(question, thread_config)
        logger.info(f"Bot answer received: {answer[:200]}...")
        
        # Increment message count for tracking
        thread_manager.increment_message_count(thread_id)
        
        await send_slack_message(channel_id, f"<@{user_id}> {answer}")
        logger.info("Slack message sent successfully")
    except Exception as e:
        logger.error(f"Error in slash command processing: {e}")
        await send_slack_message(channel_id, f"<@{user_id}> Error: {str(e)}")

async def process_slack_message(event):
    """Process Slack message/mention asynchronously using LangGraph thread management"""
    text = event.get('text', '')
    channel = event.get('channel')
    user = event.get('user')
    thread_ts = event.get('thread_ts') or event.get('ts')
    
    logger.info(f"Processing message from user {user} in channel {channel}: '{text}' (thread: {thread_ts})")
    
    # Remove bot mention from text if it's an app mention
    original_text = text
    if event.get('type') == 'app_mention':
        # Remove the bot mention (everything before the first space)
        if ' ' in text:
            text = text.split(' ', 1)[1]
            logger.info(f"Cleaned text after removing mention: '{text}'")
    
    clean_text = text.strip()
    text_lower = clean_text.lower()
    
    # Handle different commands
    if 'help' in text_lower:
        response = "Available commands:\n• Ask any Kubernetes question\n• help - Show this help message\n• stats - Show thread statistics"
        logger.info("Providing help response")
    elif 'stats' in text_lower:
        try:
            stats = thread_manager.get_stats()
            response = f"📊 **Thread Statistics:**\n"
            response += f"• Active threads: {stats['total_threads']}\n"
            response += f"• Max threads: {stats['max_threads']}\n"
            response += f"• TTL: {stats['ttl_hours']} hours\n"
            
            if stats['threads']:
                response += f"\n**Recent threads:**\n"
                for thread in stats['threads'][-5:]:  # Show last 5 threads
                    response += f"• Thread {thread['thread_id'][:8]}... ({thread['message_count']} msgs, {thread['age_hours']:.1f}h ago)\n"
            
            logger.info("Providing stats response")
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            response = f"Error getting statistics: {str(e)}"
    elif clean_text:
        try:
            # Get thread configuration for LangGraph
            thread_config = thread_manager.get_thread_config(thread_ts, channel)
            
            logger.info("Processing question with LangGraph thread management...")
            response = await bot.process_question_with_thread(clean_text, thread_config)
            logger.info(f"Bot response received: {response[:200]}...")
            
            # Increment message count for tracking
            thread_manager.increment_message_count(thread_ts)
            
        except Exception as e:
            logger.error(f"Error getting bot response: {e}")
            response = f"Error processing your question: {str(e)}"
    else:
        response = "Please ask me a Kubernetes question!"
        logger.info("Empty question, asking for input")
    
    # Send response
    logger.info("Sending response to Slack...")
    await send_slack_message(channel, response, thread_ts)
    logger.info("Message processing completed")

async def send_slack_message(channel, text, thread_ts=None):
    """Send a formatted message to Slack using slack-sdk"""
    try:
        from slack_sdk import WebClient
        from slack_sdk.models.blocks import SectionBlock, DividerBlock
        from slack_sdk.models.block_elements import MarkdownTextObject
    except ImportError:
        logger.error("slack-sdk not installed. Please run: pip install slack-sdk")
        # Fallback to simple message
        await send_slack_message_simple(channel, text, thread_ts)
        return
    
    slack_token = os.environ.get('SLACK_BOT_TOKEN')
    if not slack_token:
        logger.error("SLACK_BOT_TOKEN not set - cannot send message")
        return
    
    client = WebClient(token=slack_token)
    
    # Format text for better Slack display
    formatted_blocks = format_text_with_slack_sdk(text)
    
    try:
        response = client.chat_postMessage(
            channel=channel,
            text=text,  # Fallback text for notifications
            blocks=formatted_blocks,
            thread_ts=thread_ts
        )
        
        if response["ok"]:
            logger.info("Message sent successfully using slack-sdk")
        else:
            logger.error(f"Error sending message: {response['error']}")
            
    except Exception as e:
        logger.error(f"Exception sending message with slack-sdk: {e}")
        # Fallback to simple message
        await send_slack_message_simple(channel, text, thread_ts)

async def send_slack_message_simple(channel, text, thread_ts=None):
    """Fallback simple message sender without slack-sdk"""
    import requests
    
    slack_token = os.environ.get('SLACK_BOT_TOKEN')
    if not slack_token:
        logger.error("SLACK_BOT_TOKEN not set - cannot send message")
        return
    
    url = 'https://slack.com/api/chat.postMessage'
    headers = {
        'Authorization': f'Bearer {slack_token}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'channel': channel,
        'text': text
    }
    
    if thread_ts:
        payload['thread_ts'] = thread_ts
    
    logger.info(f"Sending simple message to channel {channel}: {text[:100]}...")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            logger.info("Simple message sent successfully")
        else:
            logger.error(f"Error sending simple message: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Exception sending simple message: {e}")

def format_text_with_slack_sdk(text):
    """Universal text formatter for Slack using slack-sdk"""
    try:
        from slack_sdk.models.blocks import SectionBlock, DividerBlock
        from slack_sdk.models.block_elements import MarkdownTextObject
    except ImportError:
        # Return simple block format if slack-sdk not available
        return [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]
    
    blocks = []
    
    # Universal approach: Split by numbered items if they exist
    import re
    numbered_items = re.split(r'(?=\d+\.\s+\*\*)', text)
    
    if len(numbered_items) > 1:
        # This has numbered items - process them
        
        # Add intro text if exists (first item before numbered list)
        if numbered_items[0].strip():
            intro_text = format_regular_text(numbered_items[0].strip())
            blocks.append(SectionBlock(
                text=MarkdownTextObject(text=intro_text)
            ))
        
        # Process numbered items
        for i, item in enumerate(numbered_items[1:], 1):  # Skip first item
            if not item.strip():
                continue
            
            # Format any numbered item universally
            formatted_text = format_numbered_item(item.strip())
            
            blocks.append(SectionBlock(
                text=MarkdownTextObject(text=formatted_text)
            ))
            
            # Add divider between items (except last)
            if i < len(numbered_items) - 1:
                blocks.append(DividerBlock())
    
    else:
        # Handle as regular text - might be multi-paragraph
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            formatted_text = format_regular_text(paragraph.strip())
            blocks.append(SectionBlock(
                text=MarkdownTextObject(text=formatted_text)
            ))
    
    return [block.to_dict() for block in blocks]

def format_numbered_item(text):
    """Format any numbered item (pods, services, deployments, etc.) for Slack"""
    import re
    
    # Convert markdown to Slack mrkdwn
    # **Text** -> *Text*
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    
    # Format numbered item header with code block styling
    text = re.sub(r'^(\d+)\.\s+\*(.*?)\*', r'`\1.` *\2*', text, flags=re.MULTILINE)
    
    # Format key-value pairs with bullet points
    text = re.sub(r'^\s*-\s+\*(.*?)\*:\s*(.*)', r'   • *\1:* \2', text, flags=re.MULTILINE)
    
    # Format simple bullet points
    text = re.sub(r'^\s*-\s+(.*)', r'   • \1', text, flags=re.MULTILINE)
    
    return text

def format_regular_text(text):
    """Format regular text for Slack"""
    import re
    
    # Convert markdown to Slack mrkdwn
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)  # Bold
    text = re.sub(r'^#+\s*(.*)', r'*\1*', text, flags=re.MULTILINE)  # Headers
    
    return text

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True) 