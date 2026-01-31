# Movie Recommender

Prosty system rekomendacji filmow. Użytkownik opisuje co chciałby obejrzeć, a system znajduje filmy zgodne z oczekiwaniami użytkownika. System obsługuje podstawowe informacje o filmie, a do tego z pomocą modelu sentence trasnformer dopasowuje słowa użytkownika do podobnych znaczeniowo słów w opisie filmu.

## Uruchomienie
Wymagany Docker Engine
```
docker build -t movie-recommender .
docker run -p 5000:5000 movie-recommender
```

Aplikacja dostępna na http://localhost:5000

## Dokumentacja

### Endpointy
- `GET /` — czyści sesję i renderuje UI.  
- `POST /chat` — przyjmuje JSON `{ "message": "..." }` i zwraca `{ "response": "..." }`.

Przykład:
```bash
curl -s http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"good thriller from the 90s with Brad Pitt"}'
```

### Flow aplikacji (high level)
1. UI wysyła tekst do `POST /chat`.
2. `app.py` trzyma kontekst rozmowy w sesji (`session["context"]`) i woła `MovieEngine.respond(...)`.
3. `MovieEngine` parsuje zapytanie → filtruje bazę → (opcjonalnie) robi ranking semantyczny → formatuje odpowiedź.

### Format danych
`imdb_movies.json` to lista filmów. Każdy rekord ma m.in. pola:
`id`, `title`, `type`, `year`, `runtime`, `genres`, `rating`, `votes`, `cast`, `overview`.

### Jak działa wyszukiwanie
**1) Parsowanie (reguły)**
- gatunek: mapowanie słów typu `horror`, `thriller`, `romance` → nazwy gatunków,
- dekady/lata: `80s/90s/2000s` albo konkretne `1994`,
- oceny: słowa typu `great/best/terrible` → zakres ratingu,
- aktor: dopasowanie nazwisk z castu w tekście,
- keywords: ekstrahowane słowa + proste rozwijanie przez słowniki synonimów.

**2) Filtry twarde**
Jeśli podasz `actor/genre/year/rating`, to filmy niespełniające warunku odpadają od razu.

**3) Ranking**
- Tryb domyślny: semantyczny (SentenceTransformer) — liczy podobieństwo kosinusowe zapytania do embeddingów filmów i bierze top wyniki.
- Fallback: dopasowanie słów kluczowych w `title/overview` z prostym scoringiem.

**4) “Humanizacja” wyników**
Silnik dodaje mały losowy “szum” w topce, żeby wyniki nie były zawsze identyczne przy podobnych zapytaniach.

### Cache embeddingów
Embeddingi są zapisywane do `imdb_embeddings.pkl`. Jeśli cache nie pasuje rozmiarem do bazy, jest przebudowywany automatycznie.
 


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
