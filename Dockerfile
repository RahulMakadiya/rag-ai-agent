FROM python:3.10

WORKDIR /app

COPY requirements-docker.txt .

# Install torch from PyPI (the +cpu variant is Windows-only and not needed in Docker/Linux)
RUN pip install torch torchaudio torchvision

# Install remaining app dependencies
RUN pip install -r requirements-docker.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]