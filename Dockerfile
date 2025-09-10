# Pre-download model để tránh timeout khi runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Railway-friendly configs  
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface
ENV RAILWAY_STARTUP_TIMEOUT=300