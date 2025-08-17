# Mindfulness Therapist

A voice-based AI therapist application that provides mindfulness guidance and cognitive behavioral therapy (CBT) through natural conversation. Built with Python, Flask, and IBM WatsonX AI.

## ✨ Features

- 🧘 **Mindfulness Guidance**: Personalized mindfulness exercises and meditation sessions
- 🎙️ **Voice Interaction**: Natural voice conversations with real-time emotion analysis
- 🧠 **CBT Techniques**: Evidence-based cognitive behavioral therapy approaches
- 🔒 **Secure & Private**: Your conversations stay on your device
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🎨 **Modern UI**: Clean interface with soothing wave visualization

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Saumik17rkl/Mindfullness_Therapist.git
   cd Mindfullness_Therapist
   ```

2. **Set up the environment**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Update with your WatsonX API credentials

4. **Run the application**
   ```bash
   python cbt.py
   ```

5. **Open in browser**
   Visit `http://localhost:5000` in your web browser

## 🛠️ Configuration

Configure the application by setting these environment variables in your `.env` file:

```
FLASK_SECRET=your_secret_key
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_URL=your_watsonx_service_url
WATSONX_PROJECT_ID=your_watsonx_project_id
```

## 📚 How It Works

The Mindfulness Therapist uses AI to:
- Analyze emotional states during conversations
- Provide evidence-based mindfulness exercises
- Guide meditation and breathing techniques
- Offer personalized CBT-based support
- Track your mindfulness journey

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

- Built with Flask and WatsonX AI
- Inspired by mindfulness and cognitive behavioral therapy principles
- Wave visualization using HTML5 Canvas