from flask import Flask, render_template, request, jsonify
import pickle
import datetime
import webbrowser
import random

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Store notes
notes = []

# Jokes list
jokes = [
    "Why do programmers prefer dark mode? Because light attracts bugs!",
    "Why did the programmer quit? Because he didn't get arrays!",
    "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
    "Why do Java developers wear glasses? Because they don't C#!",
    "A SQL query walks into a bar and asks two tables to join him!",
    "Why was the JavaScript developer sad? Because he didn't know how to null his feelings!",
    "What do you call a programmer from Finland? Nerdic!",
    "Why did the developer go broke? Because he used up all his cache!",
]

# Website URLs
websites = {
    'google': 'https://www.google.com',
    'youtube': 'https://www.youtube.com',
    'github': 'https://www.github.com',
    'facebook': 'https://www.facebook.com',
    'twitter': 'https://www.twitter.com',
    'instagram': 'https://www.instagram.com',
    'linkedin': 'https://www.linkedin.com',
    'netflix': 'https://www.netflix.com',
    'amazon': 'https://www.amazon.com',
    'wikipedia': 'https://www.wikipedia.org',
}

def process_command(text):
    text_lower = text.lower().strip()

    # Predict intent
    vec = vectorizer.transform([text_lower])
    intent = model.predict(vec)[0]
    confidence = max(model.predict_proba(vec)[0]) * 100

    response = ""
    action = None
    action_url = None

    if intent == 'greeting':
        responses = [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! Great to hear from you!",
            "Greetings! How may I assist you?"
        ]
        response = random.choice(responses)

    elif intent == 'farewell':
        responses = [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! Come back anytime!",
            "Farewell! It was nice talking to you!"
        ]
        response = random.choice(responses)

    elif intent == 'time':
        now = datetime.datetime.now()
        response = f"The current time is {now.strftime('%I:%M %p')}"

    elif intent == 'date':
        now = datetime.datetime.now()
        response = f"Today is {now.strftime('%A, %B %d, %Y')}"

    elif intent == 'search':
        query = text_lower.replace('search for', '').replace('search', '').replace('look up', '').replace('find', '').strip()
        action = 'search'
        action_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        response = f"Searching Google for: {query}"

    elif intent == 'open_website':
        site = None
        for key in websites:
            if key in text_lower:
                site = key
                break
        if site:
            action = 'open'
            action_url = websites[site]
            response = f"Opening {site.capitalize()} for you!"
        else:
            response = "Which website would you like to open?"

    elif intent == 'take_note':
        note_text = text_lower.replace('take note', '').replace('note down', '').replace('remember', '').replace('make a note', '').replace('write down', '').strip()
        notes.append({
            'text': note_text,
            'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        response = f"Note saved: {note_text}"

    elif intent == 'weather':
        response = "I can't check live weather right now, but you can say 'open google' and search for weather!"

    elif intent == 'joke':
        response = random.choice(jokes)

    elif intent == 'help':
        response = """Here's what I can do:
        🗣️ Greet you
        ⏰ Tell the time and date
        🔍 Search Google
        🌐 Open websites
        📝 Take notes
        😂 Tell jokes
        🌤️ Weather info
        Just speak or type your command!"""

    else:
        response = "I'm not sure I understood that. Try saying 'help' to see what I can do!"

    return {
        'intent': intent,
        'confidence': round(confidence, 2),
        'response': response,
        'action': action,
        'action_url': action_url
    }

@app.route('/')
def index():
    return render_template('index.html', notes=notes)

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'})
    result = process_command(text)
    return jsonify(result)

@app.route('/notes')
def get_notes():
    return jsonify(notes)

@app.route('/clear_notes', methods=['POST'])
def clear_notes():
    notes.clear()
    return jsonify({'message': 'Notes cleared!'})

@app.route('/metrics')
def show_metrics():
    return render_template('metrics.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True) 
