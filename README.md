# IPT (Interpersonal Psychotherapy) Therapist

A voice-based AI therapist application that provides Interpersonal Psychotherapy (IPT) guidance through natural conversation. Built with Python, Flask, and IBM WatsonX AI.

## ✨ Features
- 🎯 **Interpersonal Focus**: Specialized in helping with relationship issues and life transitions
- 🎙️ **Voice Interaction**: Natural voice conversations with real-time analysis
- � **Relationship Support**: Guidance for improving interpersonal relationships
- 🔒 **Secure & Private**: Your conversations stay on your device
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🎨 **Modern UI**: Clean and intuitive interface

## 🚀 Quick Start
1. Clone the repository
```bash
git clone https://github.com/Saumik17rkl/IPT.git
cd IPT
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
   - Update with your WatsonX API credentials

4. Run the application
```bash
python cbt.py
```
5. Open in browser: Visit http://localhost:5000

## 🛠️ Configuration
Configure the application by setting these environment variables in your `.env` file:
```
FLASK_SECRET=your_secret_key
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_URL=your_watsonx_service_url
WATSONX_PROJECT_ID=your_watsonx_project_id
```

## 📚 How It Works
The IPT Therapist uses AI to:
- Help identify interpersonal issues
- Provide evidence-based IPT techniques
- Guide through relationship challenges
- Support personal growth and communication
- Track progress over time

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Built with Flask and WatsonX AI
- Based on Interpersonal Psychotherapy principles
- Developed with ❤️ for mental health support
