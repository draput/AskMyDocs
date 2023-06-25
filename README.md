# Ask My Docks

A Streamlit / LangChain based WebApp for experimenting with ALM Models

Clone the repo and cd into the cloned repo directory.

## Run locally

`streamlit run app/app.py`

You can set an environment variable `OPENAI_API_KEY=sk-...` to avoid being asked to enter it for each session.

## Run with Docker Compose

This is the way!

- `docker compose up` *- add -d for deamon mode*
- `docker compose up --build`  - to rebuild an image before running it (use this after a git pull)

## Run with Docker

- `docker build . -t ask-my-docs
docker run -p 8501:8501 -v /app/db:/db ask-my-docs` - to enter the OpenAI API key manually, once per session

or

- `docker run -p 8501:8501 -v /app/db:/db --env-file .env ask-my-docs` - to use the an OpenAI API key from a .env file. In this case the OpenAI Key widget won't appear on the UI.

Your .env file should contain the OpenAI API key in a variable `OPENAI_API_KEY=sk-...`

*NOTE: The API key entered in the app's widget has a session scope. By using the .env file method, the entered API key (and the costs associated with it) will be made available to all users.*
