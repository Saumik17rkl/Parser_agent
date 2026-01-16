# Math Parser Agent

A robust mathematical problem parser that extracts structured information from natural language math problems. The application uses multiple AI providers (Groq, OpenAI, and local Mistral) with smart fallback mechanisms to ensure reliable performance.

## ✨ Features
- 🧮 **Math Problem Parsing**: Extracts variables, constraints, and problem type from natural language
- 🤖 **Multi-Model Support**: Uses Groq, OpenAI, and local Mistral models with automatic fallback
- ⚡ **High Performance**: Caching and optimized retry logic for reliability
- 🔄 **Session Management**: Tracks user sessions and history
- 🛠️ **REST API**: Easy integration with other applications
- 🚀 **Production Ready**: Ready for deployment with Gunicorn and environment-based configuration

## 🚀 Quick Start

1. Clone the repository
```bash
git clone https://github.com/Saumik17rkl/Parser_agent.git
cd Parser_agent
```

2. Set up the environment
```bash
# Create a virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. Configure environment variables
   - Copy `.env.example` to `.env`
   - Update with your API credentials

4. Run the application
```bash
python bot.py
```

5. Open in browser: Visit http://localhost:5000

## 🛠️ Configuration
Configure the application by setting these environment variables in your `.env` file:
```
FLASK_SECRET=your_secret_key
GROQ_API_KEY=your_groq_api_key
```

## 📚 How It Works
SattvaBOT uses advanced AI to:
- Provide empathetic and supportive conversations
- Detect emotional states and respond appropriately
- Offer mental health resources and coping strategies
- Maintain conversation context for personalized support
- Provide crisis intervention when needed

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Built with Flask and Groq
- Developed with ❤️ for mental health support# SattvaBOT
