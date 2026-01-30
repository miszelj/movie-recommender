FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user ./templates/ templates/
COPY --chown=user ./imdb_movies.json .
COPY --chown=user ./imdb_embeddings.pkl .
COPY --chown=user ./movie_engine.py .
COPY --chown=user ./app.py .

EXPOSE 7860

CMD ["python", "-u", "app.py"]
