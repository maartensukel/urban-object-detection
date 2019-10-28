FROM python:3.7-slim


ADD requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

cmd python detector_garb.py -i samples -o output --no-show