# Docker File - Sentiment Analysis.

FROM python:3
EXPOSE 5000
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["flaskapp.py"]
