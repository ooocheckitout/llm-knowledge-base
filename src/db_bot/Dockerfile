FROM python:3.11 AS python
WORKDIR /usr/app

COPY src/db_bot/requirements.txt src/db_bot/requirements.txt
RUN pip3 install -r src/db_bot/requirements.txt

COPY src/db_bot/requirements.tools.txt src/db_bot/requirements.tools.txt
RUN pip3 install -r src/db_bot/requirements.tools.txt

RUN playwright install --with-deps

COPY . .
RUN chmod +x src/db_bot/main.py
CMD ["python3", "-m", "src.db_bot.main"]