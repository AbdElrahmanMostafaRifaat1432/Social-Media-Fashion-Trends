FROM python:3.11
EXPOSE 8080

Add requirements.txt requirements.txt
Run pip install -r requirements.txt

WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "FashionApp.py", "--server.port=8080", "--server.address=0.0.0.0"]