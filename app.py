from flask import Flask, render_template, request, jsonify, session
from movie_engine import MovieEngine
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-prod')

DATA_PATH = os.path.join(os.path.dirname(__file__), 'imdb_movies.json')
engine = MovieEngine(DATA_PATH)


@app.route('/')
def index():
    session.clear()
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'response': 'Please type something!'})

    context = session.get('context', {})
    response, new_context = engine.respond(message, context)
    session['context'] = new_context
    return jsonify({'response': response})


if __name__ == '__main__':
    print("=" * 40)
    print("movie recommender")
    print(f"loaded {len(engine.filmy)} movies")
    print("=" * 40)
    print("http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)
