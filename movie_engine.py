import json
import re
import os
import random
import math


class MovieEngine:

    GENRES = {
        'horror': 'Horror', 'scary': 'Horror', 'terrifying': 'Horror', 'creepy': 'Horror',
        'spooky': 'Horror', 'frightening': 'Horror', 'nightmare': 'Horror', 'haunted': 'Horror',
        'thriller': 'Thriller', 'suspense': 'Thriller', 'tense': 'Thriller', 'suspenseful': 'Thriller',
        'gripping': 'Thriller', 'intense': 'Thriller',
        'comedy': 'Comedy', 'funny': 'Comedy', 'hilarious': 'Comedy', 'laugh': 'Comedy',
        'humorous': 'Comedy', 'witty': 'Comedy', 'silly': 'Comedy',
        'romance': 'Romance', 'romantic': 'Romance', 'love story': 'Romance', 'heartfelt': 'Romance',
        'action': 'Action', 'explosive': 'Action', 'fighting': 'Action', 'adrenaline': 'Action',
        'exciting': 'Action', 'stunts': 'Action',
        'sci-fi': 'Science Fiction', 'scifi': 'Science Fiction', 'science fiction': 'Science Fiction',
        'futuristic': 'Science Fiction', 'space': 'Science Fiction', 'aliens': 'Science Fiction',
        'fantasy': 'Fantasy', 'magical': 'Fantasy', 'mythical': 'Fantasy', 'enchanted': 'Fantasy',
        'animation': 'Animation', 'animated': 'Animation', 'cartoon': 'Animation',
        'drama': 'Drama', 'dramatic': 'Drama', 'emotional': 'Drama', 'tearjerker': 'Drama',
        'moving': 'Drama', 'heartbreaking': 'Drama',
        'crime': 'Crime', 'criminal': 'Crime', 'heist': 'Crime', 'gangster': 'Crime', 'mafia': 'Crime',
        'detective': 'Crime', 'spy': 'Crime',
        'war': 'War', 'military': 'War', 'battle': 'War', 'soldier': 'War',
        'western': 'Western', 'cowboy': 'Western',
        'documentary': 'Documentary', 'docuseries': 'Documentary',
        'family': 'Family', 'kids': 'Family', 'children': 'Family',
        'music': 'Music', 'musical': 'Music', 'singing': 'Music', 'concert': 'Music',
        'mystery': 'Mystery', 'mysterious': 'Mystery', 'whodunit': 'Mystery', 'enigma': 'Mystery',
        'history': 'History', 'historical': 'History', 'period': 'History', 'ancient': 'History',
        'adventure': 'Adventure', 'adventurous': 'Adventure', 'expedition': 'Adventure', 'quest': 'Adventure',
    }

    RATINGS = {
        'good': (6.5, 10), 'great': (7.5, 10), 'best': (8.0, 10),
        'excellent': (7.5, 10), 'amazing': (7.5, 10), 'classic': (7.5, 10),
        'masterpiece': (8.5, 10), 'top': (8.0, 10),
        'bad': (0, 5.0), 'terrible': (0, 4.0), 'awful': (0, 4.0),
        'horrible': (0, 4.0), 'worst': (0, 4.5), 'trash': (0, 4.5),
        'garbage': (0, 4.0), 'crap': (0, 4.5),
        'average': (5.0, 6.5), 'mediocre': (4.5, 6.0), 'okay': (5.0, 6.5), 'decent': (5.5, 7.0),
        'underrated': (5.5, 7.5),
        'so bad its good': (2.0, 5.0), 'guilty pleasure': (4.0, 6.0), 'b-movie': (3.0, 5.5),
    }

    POPULARITY = {
        'niche': (500, 50000),
        'obscure': (100, 10000),
        'hidden gem': (500, 30000),
        'underrated': (1000, 50000),
        'unknown': (100, 5000),
        'popular': (100000, None),
        'blockbuster': (500000, None),
    }

    DECADES = [
        (r'\b50s\b|\b1950s?\b|fifties', 1950, 1959),
        (r'\b60s\b|\b1960s?\b|sixties', 1960, 1969),
        (r'\b70s\b|\b1970s?\b|seventies', 1970, 1979),
        (r'\b80s\b|\b1980s?\b|eighties', 1980, 1989),
        (r'\b90s\b|\b1990s?\b|nineties', 1990, 1999),
        (r'\b2000s\b|two thousands', 2000, 2009),
        (r'\b2010s\b|twenty tens', 2010, 2019),
        (r'\b2020s\b|twenty twenties', 2020, 2029),
    ]

    GREETINGS = [
        "Hi! Tell me what kind of movie you want.",
        "Hello! What movie are you in the mood for?",
        "Hey! Describe what you'd like to watch.",
        "Hi there! Looking for a movie? Tell me what you like.",
    ]

    GENRE_QUESTIONS = [
        "What genre are you interested in? Horror, comedy, thriller, sci-fi, drama...?",
        "Tell me a genre - horror, comedy, action, drama?",
        "What type of movie? Comedy, thriller, horror...?",
    ]

    DECADE_QUESTIONS = [
        "What era? 80s, 90s, 2000s, or modern?",
        "From what decade? 80s, 90s, 2000s...?",
        "Old or new? Give me a decade or year.",
    ]

    SUCCESS_RESPONSES = [
        "Here's what I found:",
        "Check these out:",
        "I think you'll like these:",
        "Found some good ones:",
    ]

    FOLLOWUP_QUESTIONS = [
        "Want more suggestions?",
        "Do you find any of these interesting?",
        "Anything else you're looking for?",
        "Want me to find more?",
    ]
    GOODBYES = [
        "You're welcome! Enjoy watching!",
        "Have fun! Great choice!",
        "Enjoy the movie!",
        "Happy watching!",
    ]
    SYNONYMS = {
        'space': ['space', 'astronaut', 'nasa', 'spaceship', 'planet', 'galaxy', 'moon', 'orbit'],
        'robot': ['robot', 'android', 'cyborg', 'artificial', 'machine', 'automation'],
        'love': ['love', 'romance', 'romantic', 'relationship', 'heart'],
        'murder': ['murder', 'killer', 'death', 'crime', 'detective'],
        'revenge': ['revenge', 'vengeance', 'avenge', 'payback'],
        'war': ['war', 'battle', 'soldier', 'military', 'army', 'combat'],

        'money': ['money', 'rich', 'wealth', 'millionaire', 'fortune'],
        'family': ['family', 'father', 'mother', 'son', 'daughter', 'parent'],
        'friends': ['friend', 'friendship', 'buddy', 'companion'],
        'school': ['school', 'student', 'teacher', 'college', 'university', 'campus'],
        'monster': ['monster', 'creature', 'beast', 'alien'],
        'zombie': ['zombie', 'undead', 'apocalypse', 'walking dead', 'outbreak'],
        'vampire': ['vampire', 'blood', 'dracula', 'undead'],
        'magic': ['magic', 'wizard', 'witch', 'spell', 'sorcerer'],
        'cars': ['car', 'racing', 'driver', 'race', 'speed'],
        'sports': ['sports', 'team', 'player', 'coach', 'championship', 'game'],
        'music': ['music', 'band', 'singer', 'song', 'concert'],
        'dance': ['dance', 'dancing', 'dancer', 'ballet'],
        'dog': ['dog', 'puppy', 'canine', 'pet'],
        'cat': ['cat', 'kitten', 'feline', 'pet'],

        'horse': ['horse', 'riding', 'equestrian', 'stallion'],
        'dinosaur': ['dinosaur', 'prehistoric', 'jurassic', 'rex'],
        'superhero': ['superhero', 'hero', 'powers', 'villain', 'marvel', 'dc'],
        'hacker': ['hacker', 'computer', 'cyber', 'internet', 'code'],
        'island': ['island', 'stranded', 'deserted', 'tropical'],
        'ocean': ['ocean', 'sea', 'underwater', 'ship', 'boat', 'shark'],
        'plane': ['plane', 'airplane', 'flight', 'pilot', 'crash'],
        'prison': ['prison', 'jail', 'inmate', 'escape', 'guard'],
        'spy': ['spy', 'agent', 'secret', 'intelligence', 'undercover'],
        'apocalypse': ['apocalypse', 'survival', 'disaster', 'extinction'],
    }

    def __init__(self, data_path: str, use_semantic: bool = True):
        print(f"Loading movies from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.filmy = json.load(f)
        print(f"Loaded {len(self.filmy)} movies")

        self._build_actor_index()

        self.semantic = None
        if use_semantic:
            try:
                self._init_semantic_search(data_path)
            except Exception as e:
                print(f"Semantic search disabled: {e}")
                self.semantic = False

    def _build_actor_index(self):
        self.actors = {}
        for film in self.filmy:
            for actor in film.get('cast', []):
                name_norm = actor.lower().rstrip('.')
                self.actors[name_norm] = actor.lower()
        print(f"Indexed {len(self.actors)} actors")

    def _init_semantic_search(self, data_path: str):
        import pickle
        import numpy as np

        cache_path = os.path.join(os.path.dirname(data_path), 'imdb_embeddings.pkl')

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                if len(self.embeddings) == len(self.filmy):
                    print(f"Loaded embeddings from cache ({len(self.embeddings)} movies)")
                    self._load_embedding_model()
                    self._build_genre_embeddings()
                    self.semantic = True
                    return
            except Exception as e:
                print(f"Cache invalid: {e}")

        print("Building embeddings (first run only, please wait)...")
        self._load_embedding_model()

        texts = []
        for m in self.filmy:
            parts = [m['title']]
            if m.get('genres'):
                parts.append(', '.join(m['genres']))
            if m.get('cast'):
                parts.append('starring ' + ', '.join(m['cast'][:5]))
            if m.get('overview'):
                parts.append(m['overview'][:500])
            texts.append(' | '.join(parts))

        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"Saved embeddings to {cache_path}")
        self._build_genre_embeddings()
        self.semantic = True

    def _build_genre_embeddings(self):
        import numpy as np
        all_genres = set()
        for film in self.filmy:
            for g in film.get('genres', []):
                all_genres.add(g)
        self.genre_list = sorted(all_genres)
        self.genre_embeddings = self.model.encode(self.genre_list, convert_to_numpy=True)
        print(f"Built embeddings for {len(self.genre_list)} genres")

    def _match_genre_semantic(self, text: str, threshold: float = 0.35) -> str:
        import numpy as np
        query_embedding = self.model.encode(text, convert_to_numpy=True)
        similarities = np.dot(self.genre_embeddings, query_embedding) / (
            np.linalg.norm(self.genre_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )
        best_idx = np.argmax(similarities)
        if similarities[best_idx] >= threshold:
            return self.genre_list[best_idx]
        return None

    def _load_embedding_model(self):
        if not hasattr(self, 'model') or self.model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _find_actor_in_query(self, text: str) -> str:
        text_lower = text.lower().rstrip('.')
        sorted_actors = sorted([a for a in self.actors.keys() if len(a) >= 3], key=len, reverse=True)
        for actor_norm in sorted_actors:
            pattern = r'\b' + re.escape(actor_norm) + r'\b'
            if re.search(pattern, text_lower):
                return self.actors[actor_norm]
        return None

    def parse_query(self, text: str) -> dict:
        text_lower = text.lower()
        result = {
            'genre': None,
            'year_from': None,
            'year_to': None,
            'rating_min': None,
            'rating_max': None,
            'votes_min': None,
            'votes_max': None,
            'actor': None,
            'keywords': [],
        }

        result['actor'] = self._find_actor_in_query(text)

        for word, genre in self.GENRES.items():
            if word in text_lower:
                result['genre'] = genre
                break


        for pattern, year_from, year_to in self.DECADES:
            if re.search(pattern, text_lower):
                result['year_from'] = year_from
                result['year_to'] = year_to
                break

        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if year_match and result['year_from'] is None:
            year = int(year_match.group(1))
            result['year_from'] = year
            result['year_to'] = year

        sorted_ratings = sorted(self.RATINGS.items(), key=lambda x: len(x[0]), reverse=True)
        for word, (rating_min, rating_max) in sorted_ratings:
            if ' ' in word:
                if word in text_lower:
                    result['rating_min'] = rating_min
                    result['rating_max'] = rating_max
                    break
            else:
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, text_lower):
                    result['rating_min'] = rating_min
                    result['rating_max'] = rating_max
                    break

        sorted_popularity = sorted(self.POPULARITY.items(), key=lambda x: len(x[0]), reverse=True)
        for word, (votes_min, votes_max) in sorted_popularity:
            if ' ' in word:
                if word in text_lower:
                    result['votes_min'] = votes_min
                    result['votes_max'] = votes_max
                    break
            else:
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, text_lower):
                    result['votes_min'] = votes_min
                    result['votes_max'] = votes_max
                    break

        stop_words = {'a', 'an', 'the', 'i', 'o', 'w', 'z', 'na', 'do', 'to',
                      'film', 'movie', 'movies', 'about', 'with', 'from'}

        if result['actor']:
            for part in result['actor'].split():
                stop_words.add(part.lower())

        for rating_word in self.RATINGS.keys():
            for part in rating_word.split():
                stop_words.add(part.lower())

        for pop_word in self.POPULARITY.keys():
            for part in pop_word.split():
                stop_words.add(part.lower())

        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        base_words = [w for w in words if len(w) > 3 and w not in stop_words
                      and w not in self.GENRES and w not in self.RATINGS]

        expanded_words = []
        for word in base_words:
            found = False
            for key, synonyms in self.SYNONYMS.items():
                if word.startswith(key[:4]) or key.startswith(word[:4]):
                    expanded_words.extend(synonyms)
                    found = True
                    break
            if not found:
                expanded_words.append(word)

        seen = set()
        result['keywords'] = []
        for w in expanded_words:
            if w not in seen:
                seen.add(w)
                result['keywords'].append(w)

        return result

    def search_movies(self, criteria: dict, limit: int = 5, raw_query: str = None) -> list:
        import numpy as np

        def matches(film):
            if criteria.get('actor'):
                cast_lower = [a.lower() for a in film.get('cast', [])]
                if criteria['actor'] not in cast_lower:
                    return False
            if criteria['genre']:
                if criteria['genre'] not in film.get('genres', []):
                    return False
            year = film.get('year')
            if criteria['year_from'] and year:
                if year < criteria['year_from'] or year > criteria['year_to']:
                    return False
            rating = film.get('rating', 0)
            if criteria['rating_min'] is not None:
                if rating < criteria['rating_min'] or rating > criteria['rating_max']:
                    return False
            votes = film.get('votes', 0)
            if criteria['votes_min'] is not None:
                if votes < criteria['votes_min']:
                    return False
            if criteria['votes_max'] is not None:
                if votes > criteria['votes_max']:
                    return False
            return True

        if self.semantic and raw_query:
            query_embedding = self.model.encode(raw_query, convert_to_numpy=True)
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
            )
            candidates = []
            for idx in np.argsort(similarities)[::-1]:
                film = self.filmy[idx]
                if matches(film):
                    candidates.append((similarities[idx], film))
                    if len(candidates) >= 50:
                        break
        else:
            candidates = []
            for film in self.filmy:
                if not matches(film):
                    continue
                desc = film.get('overview', '').lower()
                title = film.get('title', '').lower()
                hits = sum(1 for w in criteria['keywords'] if w in desc or w in title)
                if criteria['keywords'] and hits == 0 and not criteria.get('actor'):
                    continue
                candidates.append((hits, film))
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[:50]

        query_words = [w.lower() for w in (raw_query or '').split() if len(w) > 2]

        def weighted_score(candidate):
            similarity, film = candidate
            votes = film.get('votes', 1)
            vote_score = math.log10(votes + 1) / 7
            overview = film.get('overview', '').lower()
            title = film.get('title', '').lower()
            overview_hits = sum(1 for w in query_words if w in overview or w in title)
            overview_boost = min(overview_hits / max(len(query_words), 1), 1.0)
            return similarity * 0.5 + overview_boost * 0.4 + vote_score * 0.1

        ranked = sorted(candidates, key=weighted_score, reverse=True)
        return [film for _, film in ranked[:limit]]

    def format_movie(self, film: dict, number: int) -> str:
        title = film.get('title', 'Unknown')
        year = film.get('year', '?')
        genres = ', '.join(film.get('genres', [])[:3])
        rating = film.get('rating', 0)
        votes = film.get('votes', 0)
        cast = ', '.join(film.get('cast', [])[:3])
        plot = film.get('overview', '')
        if len(plot) > 200:
            plot = plot[:200] + '...'

        text = f"\n**{number}. {title}** ({year})\n"
        text += f"   Genre: {genres}\n"
        text += f"   Rating: {rating}/10 ({votes:,} votes)\n"
        if cast:
            text += f"   Cast: {cast}\n"
        if plot:
            text += f"   Plot: {plot}\n"
        return text

    def is_greeting(self, text: str) -> bool:
        greetings = ['hello', 'hi', 'hey', 'greetings', 'help', 'start']
        text_lower = text.lower().strip()
        return any(text_lower.startswith(g) for g in greetings) and len(text) < 20

    def is_thanks(self, text: str) -> bool:
        words = ['thanks', 'thank', 'bye', 'goodbye', 'cheers']
        return any(w in text.lower() for w in words)

    def is_random(self, text: str) -> bool:
        phrases = ['surprise me', 'random', 'anything', 'whatever', 'idk', "don't know", 'no idea', 'you choose', 'dealer']
        text_lower = text.lower()
        return any(p in text_lower for p in phrases)

    def get_random_movies(self, limit: int = 5) -> list:
        good_movies = [f for f in self.filmy if f.get('rating', 0) >= 7.0 and f.get('votes', 0) >= 10000]
        return random.sample(good_movies, min(limit, len(good_movies)))

    def respond(self, message: str, context: dict = None) -> tuple:
        if context is None:
            context = {}

        topic = context.get('topic', 'start')

        if self.is_greeting(message):
            return (
                random.choice(self.GREETINGS) + "\n\n"
                "Examples:\n"
                "- \"good thriller from the 90s\"\n"
                "- \"funny romantic comedy\"\n"
                "- \"movie about space\"",
                {'topic': 'ready'}
            )

        if self.is_thanks(message):
            return (random.choice(self.GOODBYES), {'topic': 'end'})

        if self.is_random(message):
            movies = self.get_random_movies()
            response = "Here are some random picks for you:\n"
            for i, film in enumerate(movies, 1):
                response += self.format_movie(film, i)
            response += "\n" + random.choice(self.FOLLOWUP_QUESTIONS)
            return (response, {'topic': 'showing'})

        if topic == 'showing' and self._wants_more(message):
            context['offset'] = context.get('offset', 0) + 5
            return self._search_and_respond(message, context, use_offset=True)

        if self._wants_different(message):
            return (
                random.choice(self.GENRE_QUESTIONS),
                {'topic': 'asking_genre'}
            )

        criteria = self.parse_query(message)

        has_genre = criteria['genre'] is not None
        has_keywords = len(criteria['keywords']) > 0
        has_actor = criteria.get('actor') is not None
        has_year = criteria['year_from'] is not None

        is_refinement = not has_genre and not has_keywords and (has_year or has_actor)

        if is_refinement and topic == 'showing':
            if context.get('genre') and not criteria['genre']:
                criteria['genre'] = context['genre']
            if context.get('year_from') and not criteria['year_from']:
                criteria['year_from'] = context['year_from']
                criteria['year_to'] = context['year_to']
            if context.get('actor') and not criteria['actor']:
                criteria['actor'] = context['actor']
            if context.get('keywords') and not criteria['keywords']:
                criteria['keywords'] = context['keywords']

        if not has_genre and not has_keywords and not criteria['year_from'] and not has_actor:
            if topic == 'asking_decade':
                return (random.choice(self.GENRE_QUESTIONS), {'topic': 'asking_genre'})
            return (
                "Tell me more!\n\n" + random.choice(self.GENRE_QUESTIONS) +
                "\nOr describe what the movie should be about.",
                {'topic': 'asking_genre', **context}
            )

        if has_genre and not criteria['year_from'] and topic == 'asking_genre':
            context['genre'] = criteria['genre']
            context['topic'] = 'asking_decade'
            return self._search_and_respond(message, context, criteria=criteria)

        return self._search_and_respond(message, context, criteria=criteria)

    def _wants_more(self, text: str) -> bool:
        words = ['more', 'another', 'again', 'next']
        return any(w in text.lower() for w in words) and len(text) < 30

    def _wants_different(self, text: str) -> bool:
        words = ['different', 'else', 'other', 'change', 'switch']
        return any(w in text.lower() for w in words) and len(text) < 30

    def _search_and_respond(self, message: str, context: dict, criteria: dict = None, use_offset: bool = False) -> tuple:
        if use_offset:
            criteria = {
                'genre': context.get('genre'),
                'year_from': context.get('year_from'),
                'year_to': context.get('year_to'),
                'actor': context.get('actor'),
                'keywords': context.get('keywords', []),
                'rating_min': None,
                'rating_max': None,
                'votes_min': None,
                'votes_max': None,
            }
        elif criteria is None:
            criteria = self.parse_query(message)

        offset = context.get('offset', 0) if use_offset else 0
        raw_query = context.get('last_query', message) if use_offset else message
        movies = self.search_movies(criteria, limit=5 + offset, raw_query=raw_query)

        if offset:
            movies = movies[offset:] if len(movies) > offset else []

        if not movies:
            suggestion = "I couldn't find "
            suggestion += "more movies" if use_offset else "movies matching"
            suggestion += ":\n"
            if criteria.get('actor'):
                suggestion += f"- Actor: {criteria['actor'].title()}\n"
            if criteria['genre']:
                suggestion += f"- Genre: {criteria['genre']}\n"
            if criteria['year_from']:
                suggestion += f"- Years: {criteria['year_from']}-{criteria['year_to']}\n"
            if criteria.get('keywords'):
                suggestion += f"- Keywords: {', '.join(criteria['keywords'][:5])}\n"
            suggestion += "\n" + random.choice(self.GENRE_QUESTIONS)
            return (suggestion, {'topic': 'asking_genre'})

        filters = []
        if criteria['genre']:
            filters.append(f"Genre: {criteria['genre']}")
        if criteria['year_from']:
            if criteria['year_from'] == criteria['year_to']:
                filters.append(f"Year: {criteria['year_from']}")
            else:
                filters.append(f"Years: {criteria['year_from']}-{criteria['year_to']}")
        if criteria.get('actor'):
            filters.append(f"Actor: {criteria['actor'].title()}")
        if criteria.get('rating_min') is not None:
            filters.append(f"Rating: {criteria['rating_min']}-{criteria['rating_max']}")
        if criteria.get('votes_min') is not None:
            filters.append(f"Votes: {criteria['votes_min']}-{criteria['votes_max'] or '∞'}")

        response = random.choice(self.SUCCESS_RESPONSES)
        if filters:
            response += f" [{', '.join(filters)}]"
        response += "\n"
        for i, film in enumerate(movies, 1):
            response += self.format_movie(film, i)
        response += "\n" + random.choice(self.FOLLOWUP_QUESTIONS)

        new_context = {
            'topic': 'showing',
            'genre': criteria['genre'],
            'year_from': criteria['year_from'],
            'year_to': criteria['year_to'],
            'actor': criteria.get('actor'),
            'keywords': criteria.get('keywords', []),
            'offset': offset,
            'last_query': raw_query,
        }

        return (response, new_context)
