FROM python:3.6.2

WORKDIR /app

# copy and install requirements to cache layer
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "training_script.py"]
