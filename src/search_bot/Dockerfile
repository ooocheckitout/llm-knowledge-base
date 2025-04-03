FROM python:3.11 AS python
WORKDIR /usr/app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
RUN chmod +x lileg_db_bot.py
CMD ["python3", "lileg_search_bot.py"]