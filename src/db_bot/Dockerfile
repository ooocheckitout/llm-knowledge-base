FROM python:3.11 AS python
WORKDIR /usr/app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY requirements.tools.txt .
RUN pip3 install -r requirements.tools.txt

RUN playwright install --with-deps

COPY . .
RUN chmod +x main.py
CMD ["python3", "main.py"]