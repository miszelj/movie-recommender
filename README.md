# Movie Recommender

Prosty system rekomendacji filmow. Użytkownik opisuje co chciałby obejrzeć, a system znajduje filmy zgodne z oczekiwaniami użytkownika. System obsługuje podstawowe informacje o filmie, a do tego z pomocą modelu sentence trasnformer dopasowuje słowa użytkownika do podobnych znaczeniowo słów w opisie filmu.

## Uruchomienie
Wymagany Docker Engine
```
docker build -t movie-recommender .
docker run -p 5000:5000 movie-recommender
```

Aplikacja dostępna na http://localhost:5000

## Jak to dziala

Baza ma okolo 10 tysięcy filmow z TMDB. Przy pierwszym uruchomieniu buduje embeddingi, potem juz ładuje dane z cache.

Wyszukiwanie laczy wyszukiwanie semantyczne z filtrami - gatunek, rok, aktor, ocena. Po wpisaniu "thriller from the 90s" bot pokazuje propozycje zgodne z oczekiwaniami użytkownika

## Pliki

- app.py - flask, obsluga requestow
- movie_engine.py - logika wyszukiwania
- imdb_movies.json - zbiór danych z informacjami o filmach
- imdb_embeddings.pkl - cache embedddingow

## Techniczne

- Model: all-MiniLM-L6-v2 (sentence-transformers)
- Backend: Flask
- Konteneryzacja: Docker
