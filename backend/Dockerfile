FROM python:3.11.5
WORKDIR /
COPY . .
RUN pip install python-dotenv
RUN pip install poetry
RUN poetry config virtualenvs.create false  \
   && poetry install
EXPOSE 8000
CMD ["python", "main.py"]