FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python", "model_v2.py" ]
