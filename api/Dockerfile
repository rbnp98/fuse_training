# Docker File - Sentiment Analysis.

FROM python:latest
EXPOSE 5000
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python", "flaskapp.py"]
