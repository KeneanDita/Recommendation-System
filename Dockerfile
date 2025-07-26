FROM python:3.11.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir pyarrow && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py"]
