# base image
FROM python:3.9
# copy all my files in the repo to the base image
COPY . /app
# set the working directory
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app