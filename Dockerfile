FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/logs /app/models && chmod -R 777 /app

EXPOSE 7860

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=7860", "--browser.gatherUsageStats=false"]
