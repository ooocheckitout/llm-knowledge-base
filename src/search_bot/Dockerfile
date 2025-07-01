FROM python:3.11 AS python
WORKDIR /usr/app

COPY src/search_bot/requirements.txt src/search_bot/requirements.txt
RUN pip3 install -r src/search_bot/requirements.txt

COPY src/search_bot/requirements.tools.txt src/search_bot/requirements.tools.txt
RUN pip3 install -r src/search_bot/requirements.tools.txt

COPY . .
RUN chmod +x src/search_bot/main.py
CMD ["python3", "-m", "src.search_bot.main"]