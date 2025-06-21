# 🤖 Orbix - Kubernetes AI Slack Bot

A powerful AI-powered Slack bot that helps you manage and interact with Kubernetes clusters directly from Slack. Built with LangGraph for intelligent conversation threads and MCP (Model Context Protocol) for robust Kubernetes operations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Slack](https://img.shields.io/badge/slack-bot-purple.svg)
![Kubernetes](https://img.shields.io/badge/kubernetes-ready-blue.svg)

## 🌟 Features

- **🎯 Natural Language K8s Operations**: Ask questions about your Kubernetes cluster in plain English
- **🧵 Thread Management**: Maintains conversation context using LangGraph memory
- **⚡ Real-time Responses**: Instant feedback on cluster status, pod health, and more  
- **🔐 Secure**: Uses environment variables for all secrets, Slack signature verification
- **📊 Monitoring**: Built-in health checks and thread statistics
- **🎛️ Flexible**: Supports both slash commands and @mentions
- **🚀 Production Ready**: Docker support with proper logging and error handling

## 🏗️ Architecture

```mermaid
graph TD
    A[Slack User] --> B[Slack API]
    B --> C[Orbix Bot]
    C --> D[LangGraph Agent]
    D --> E[MCP Server]
    E --> F[Kubernetes API]
    
    C --> G[Thread Manager]
    C --> H[Memory Saver]
    
    F --> I[Pods]
    F --> J[Services] 
    F --> K[Deployments]
    F --> L[ConfigMaps]
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Kubernetes cluster access
- Slack workspace with bot permissions
- OpenAI API key

### 1. Clone Repository

```bash
git clone https://github.com/orbix-labs/orbix.git
cd orbix
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file:

```bash
# Slack Configuration
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# AI Configuration  
OPENAI_API_KEY=your-openai-api-key

# MCP Server Configuration
MCP_SERVER_COMMAND=node
MCP_SERVER_ARGS=/app/mcp-server-kubernetes/dist/index.js

# Kubernetes Configuration (optional if using in-cluster config)
KUBECONFIG=/path/to/your/kubeconfig

# Server Configuration
PORT=8080
```

### 4. Set Up Slack App

1. Go to [Slack API](https://api.slack.com/apps) and create a new app
2. Add Bot Token Scopes:
   - `app_mentions:read`
   - `chat:write`
   - `commands`
3. Install app to workspace
4. Set up Event Subscriptions pointing to `https://your-domain.com/slack/events`
5. Add slash command `/k8s` pointing to the same URL

### 5. Run the Bot

```bash
python agent/slack.py
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t orbix-slack-bot .
```

### Run Container

```bash
docker run -d \
  --name orbix-bot \
  -p 8080:8080 \
  -e SLACK_BOT_TOKEN=your-token \
  -e SLACK_SIGNING_SECRET=your-secret \
  -e OPENAI_API_KEY=your-key \
  -v ~/.kube/config:/root/.kube/config:ro \
  orbix-slack-bot
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orbix-slack-bot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: orbix-slack-bot
  template:
    metadata:
      labels:
        app: orbix-slack-bot
    spec:
      containers:
      - name: orbix-bot
        image: orbix-slack-bot:latest
        ports:
        - containerPort: 8080
        env:
        - name: SLACK_BOT_TOKEN
          valueFrom:
            secretKeyRef:
              name: orbix-secrets
              key: slack-bot-token
        - name: SLACK_SIGNING_SECRET
          valueFrom:
            secretKeyRef:
              name: orbix-secrets  
              key: slack-signing-secret
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: orbix-secrets
              key: openai-api-key
```

## 💬 Usage

### Slash Commands

```bash
# Ask about pods
/k8s show me pods in default namespace

# Check deployment status  
/k8s what's the status of my nginx deployment?

# Get help
/k8s help
```

### @Mentions

```
@Orbix how many pods are running in production?

@Orbix scale my api deployment to 5 replicas

@Orbix show me recent events in kube-system
```

### Available Commands

- **Cluster Information**: pods, nodes, namespaces, services
- **Deployments**: status, scaling, rollouts
- **Monitoring**: events, logs, resource usage  
- **Troubleshooting**: pod diagnostics, network issues
- **Help & Statistics**: command help, bot statistics

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/test-mcp` | GET | Test MCP server connection |
| `/threads` | GET | Get thread statistics |
| `/slack/events` | POST | Slack events webhook |

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SLACK_BOT_TOKEN` | ✅ | Slack bot OAuth token |
| `SLACK_SIGNING_SECRET` | ✅ | Slack app signing secret |
| `OPENAI_API_KEY` | ✅ | OpenAI API key for GPT-4 |
| `MCP_SERVER_COMMAND` | ❌ | MCP server command (default: node) |
| `MCP_SERVER_ARGS` | ❌ | MCP server arguments |
| `KUBECONFIG` | ❌ | Path to kubeconfig file |
| `PORT` | ❌ | Server port (default: 8080) |

### Thread Management

- **Max Threads**: 1000 (configurable)
- **TTL**: 24 hours (configurable)  
- **Auto Cleanup**: Removes expired threads automatically
- **Memory Efficient**: Uses LangGraph's MemorySaver

## 🛠️ Development

### Project Structure

```
orbix/
├── agent/
│   └── slack.py          # Main bot application
├── requirements.txt      # Python dependencies  
├── Dockerfile           # Container configuration
├── .env.example         # Environment template
└── README.md           # This file
```

### Key Components

- **ThreadManager**: Manages conversation threads and memory
- **SlackBot**: Core bot logic and MCP integration
- **Message Processing**: Async handling of Slack events
- **Security**: Request verification and secret management

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Monitoring & Observability

### Health Checks

```bash
curl http://localhost:8080/health
```

### Thread Statistics

```bash
curl http://localhost:8080/threads
```

### Logs

The bot provides comprehensive logging:
- Request/response tracking
- Thread management events
- MCP server interactions
- Error handling and debugging

## 🔒 Security Considerations

- ✅ No hardcoded secrets
- ✅ Slack signature verification
- ✅ Environment variable configuration
- ✅ Request validation
- ⚠️ Add authentication for debug endpoints in production
- ⚠️ Use HTTPS in production
- ⚠️ Implement rate limiting

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings for public methods
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for conversation management
- [MCP](https://github.com/modelcontextprotocol) for Kubernetes integration  
- [Slack SDK](https://github.com/slackapi/python-slack-sdk) for Slack integration
- OpenAI for GPT-4 capabilities

## 📞 Support

- 🐛 [Bug Reports](https://github.com/yourusername/orbix/issues)
- 💡 [Feature Requests](https://github.com/yourusername/orbix/issues)
- 📧 Email: vanchonlee@gmail.com
- 💬 Slack: #orbix-support

---

Made with ❤️ for the Kubernetes community 