FROM python:3.11 AS python
WORKDIR /usr/app

COPY src/search_bot/requirements.txt src/search_bot/requirements.txt
RUN pip3 install -r src/search_bot/requirements.txt

COPY . .
RUN chmod +x src/search_bot/main.py
CMD ["python3", "src.search_bot.main.py"]