# -------------------- Main --------------------
if __name__ == "__main__":
    logger.info("Starting GROQ CBT Chatbot Server...")
    # Check if running in production (Render sets the PORT environment variable)
    is_production = os.environ.get('PORT') is not None
    
    if is_production:
        logger.info("Running in production mode")
        port = int(os.environ.get('PORT', 8000))
        socketio.run(app, 
                    host="0.0.0.0", 
                    port=port, 
                    debug=False, 
                    allow_unsafe_werkzeug=True)
    else:  # Running locally
        logger.info("Running in development mode")
        socketio.run(app, 
                   host="0.0.0.0", 
                   port=8000, 
                   debug=True)
