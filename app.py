from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import sqlite3
import requests
from datetime import datetime
import threading
import functools
import time
from openai import OpenAI
from cachetools import TTLCache
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from flask_dance.contrib.google import make_google_blueprint, google
from groq import Groq


# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Required for session management

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Yi-1.5 Chat LLM via vLLM OpenAI-compatible API
groq_api_key = "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"  # vLLM doesn't require a real key for local use
openai_api_base = "http://localhost:8000/v1"  # Default vLLM server address

client = Groq(
    api_key=groq_api_key,
    # base_url=openai_api_base,
)
# Cache configurations
api_cache = TTLCache(maxsize=100, ttl=600)  # Cache for 10 minutes
yi_cache = TTLCache(maxsize=50, ttl=300)  # Cache Yi responses for 5 minutes

# Database configuration
DB_PATH = 'chatbot.db'
db_lock = threading.Lock()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        # Create conversations table if not exists
        conn.execute('''CREATE TABLE IF NOT EXISTS conversations 
                     (id INTEGER PRIMARY KEY, user_input TEXT, bot_response TEXT, timestamp TEXT)''')
        
        # Create users table if not exists
        conn.execute('''CREATE TABLE IF NOT EXISTS users 
                     (id INTEGER PRIMARY KEY, email TEXT UNIQUE, password_hash TEXT, 
                      name TEXT, sso_provider TEXT, sso_id TEXT)''')
        conn.commit()

def store_conversation_async(user_input, bot_response):
    """Store conversation asynchronously to avoid blocking the response"""
    def _store():
        try:
            with db_lock, get_db_connection() as conn:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "INSERT INTO conversations (user_input, bot_response, timestamp) VALUES (?, ?, ?)",
                    (user_input, bot_response, timestamp)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
    
    # Run in a separate thread to not block the main thread
    thread = threading.Thread(target=_store)
    thread.daemon = True
    thread.start()

# Caching decorator
def cached(cache_obj):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache_obj:
                logger.info(f"Cache hit for {func.__name__}")
                return cache_obj[key]
            result = func(*args, **kwargs)
            cache_obj[key] = result
            return result
        return wrapper
    return decorator

@cached(yi_cache)
def get_yi_response(user_input):
    try:
        start_time = time.time()
        
        # Use OpenAI API for generating response
        response = client.chat.completions.create(
            model="mistral-saba-24b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to help elderly people with various queries. Be patient, clear, and supportive."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        
        # Extract the response text
        ai_response = response.choices[0].message.content.strip()
        
        elapsed = time.time() - start_time
        logger.info(f"AI response time: {elapsed:.2f}s")
        return ai_response
    except Exception as e:
        logger.error(f"AI model error: {str(e)}")
        return "Sorry, I couldn't process that request. Please try again later."


# Lottery results
def get_lottery_results():
    logger.info("Fetching lottery results")
    lottery_odds = {
        "Singapore - Toto": "6/49 - Odds: 1 in 13,983,816",
        "Singapore - 4D": "4/1000 - Odds: 1 in 10,000 (1st Prize), 1 in 1,000 (other prizes)",
        "Australia - Powerball": "7/35 + 1/20 - Odds: 1 in 134,490,400",
        "Japan - Loto 6": "6/43 - Odds: 1 in 6,096,454"
    }
    response = "Lottery Odds:\n"
    for game, odds in lottery_odds.items():
        if "Singapore" in game:
            response += f"- {game}: {odds}\n"
    return response if response.strip() else "No lottery odds available."

# Weather forecast
@cached(api_cache)
def get_weather_forecast():
    logger.info("Fetching weather forecast")
    api_key = "dda1f3f75baa354ca8ad4183fed70429"  # OpenWeatherMap API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Singapore&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"Weather in Singapore: {temp}°C, {desc}"
    except requests.exceptions.Timeout:
        logger.error("Weather API timeout")
        return "Weather information is currently unavailable. Please try again later."
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return f"Error fetching weather information. Please try again later."

# Latest news
def get_latest_news():
    logger.info("Fetching latest news")
    api_key = "f63f26bbb56049909f7426ecaca11c73"  # NewsAPI key
    url = f"https://newsapi.org/v2/top-headlines?apiKey={api_key}&language=en"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        news_items = []
        for i in range(min(3, len(data["articles"]))):
            news_items.append(data["articles"][i]["title"])
        return "Latest News:\n- " + "\n- ".join(news_items)
    except requests.exceptions.Timeout:
        logger.error("News API timeout")
        return "News information is currently unavailable. Please try again later."
    except Exception as e:
        logger.error(f"News API error: {str(e)}")
        return f"Error fetching news. Please try again later."
    
# Serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Auth status endpoint - for AJAX calls
@app.route('/auth/status', methods=['GET'])
def auth_status():
    if 'user_id' in session:
        with db_lock, get_db_connection() as conn:
            user = conn.execute(
                "SELECT id, email, name FROM users WHERE id = ?", (session['user_id'],)
            ).fetchone()
            
        if user:
            return jsonify({
                "authenticated": True,
                "user": {
                    "id": user['id'],
                    "email": user['email'],
                    "name": user['name']
                }
            })
    
    return jsonify({"authenticated": False})

# User registration
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    password_hash = generate_password_hash(password)

    try:
        with db_lock, get_db_connection() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
                (email, password_hash, name)
            )
            conn.commit()
        # Success response - client will handle redirect
        return jsonify({"message": "User registered successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already exists"}), 400
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({"error": "Registration failed"}), 500

# User login
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    with db_lock, get_db_connection() as conn:
        user = conn.execute(
            "SELECT id, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()

    if user and check_password_hash(user['password_hash'], password):
        session['user_id'] = user['id']  # Store user ID in session
        # Success response - client will handle redirect
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 401

# User logout
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)  # Remove user ID from session
    return jsonify({"message": "Logout successful"}), 200

# Google SSO setup
google_blueprint = make_google_blueprint(
    client_id="YOUR_GOOGLE_CLIENT_ID",
    client_secret="YOUR_GOOGLE_CLIENT_SECRET",
    scope=["profile", "email"],
    redirect_to="google_login_callback"
)
app.register_blueprint(google_blueprint, url_prefix="/google_login")

@app.route('/google_login')
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    return redirect(url_for("google_login_callback"))

@app.route('/google_login_callback')
def google_login_callback():
    if not google.authorized:
        return redirect(url_for('index', error='google_auth_failed'))

    resp = google.get("/oauth2/v2/userinfo")
    if resp.ok:
        user_info = resp.json()
        email = user_info['email']
        name = user_info['name']

        with db_lock, get_db_connection() as conn:
            user = conn.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()

            if not user:
                conn.execute(
                    "INSERT INTO users (email, name, sso_provider, sso_id) VALUES (?, ?, ?, ?)",
                    (email, name, "google", user_info['id'])
                )
                conn.commit()
                user = conn.execute(
                    "SELECT id FROM users WHERE email = ?", (email,)
                ).fetchone()

            session['user_id'] = user['id']
            # Redirect to home page after successful Google login
            return redirect(url_for('index', success=True))
    else:
        # Redirect to home page with error parameter
        return redirect(url_for('index', error='google_login_failed'))

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Check for specific commands
    lower_input = user_input.lower()
    
    if "lottery" in lower_input:
        response = get_lottery_results()
    elif "weather" in lower_input:
        response = get_weather_forecast()
    elif "news" in lower_input:
        response = get_latest_news()
    else:
        response = get_yi_response(user_input)

    # Store conversation asynchronously
    store_conversation_async(user_input, response)
    
    return jsonify({"response": response})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

# Initialize database and run app
if __name__ == '__main__':
    init_db()
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)