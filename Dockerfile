FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY titanic_improved.py .
COPY data ./data

CMD ["python", "titanic_improved.py"]
