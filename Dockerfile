# syntax=docker/dockerfile:1

FROM ghcr.io/imagegenius/baseimage-ubuntu:lunar

# set version label
ARG BUILD_DATE
ARG VERSION
LABEL build_version="ImageGenius version:- ${VERSION} Build-date:- ${BUILD_DATE}"
LABEL maintainer="hydazz"

COPY requirements.txt /tmp/requirements.txt

RUN \
  echo "**** add 3rd party repos ****" && \
  mkdir -p /etc/apt/keyrings && \
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
  curl -fsSL https://dl-ssl.google.com/linux/linux_signing_key.pub | \
    gpg --dearmor -o /etc/apt/keyrings/google.gpg && \
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu lunar stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null && \
  echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/google.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > \
    /etc/apt/sources.list.d/google-chrome.list && \
  echo "**** install runtime packages ****" && \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    docker-ce \
    google-chrome-stable \
    python3 \
    python3-pip \
    python3-setuptools \
    unzip && \
  echo "**** install chrome driver ****" && \
  CHROME_RELEASE=$(curl -sLk https://chromedriver.storage.googleapis.com/LATEST_RELEASE) && \
  echo "Retrieving Chrome driver version ${CHROME_RELEASE}" && \
  curl -sk -o \
    /tmp/chrome.zip -L \
    "https://chromedriver.storage.googleapis.com/${CHROME_RELEASE}/chromedriver_linux64.zip" && \
  cd /tmp && \
  unzip chrome.zip && \
  mv chromedriver /usr/bin/chromedriver && \
  chown root:root /usr/bin/chromedriver && \
  chmod +x /usr/bin/chromedriver && \
  echo "**** Install python deps ****" && \
  pip3 install --break-system-packages -U --no-cache-dir -r /tmp/requirements.txt && \
  echo "**** cleanup ****" && \
  apt-get autoclean && \
  rm -rf \
    /var/lib/apt/lists/* \
    /var/tmp/* \
    /tmp/*

# copy local files
COPY ci /ci
COPY test_build.py test_build.py

ENTRYPOINT [ "" ]
