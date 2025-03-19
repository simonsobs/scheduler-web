# Use ubuntu base image
FROM ubuntu:22.04

WORKDIR /app

# Ensure clock set to UTC
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y python3 \
    python3-pip \
    git

# Install python dependencies
COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy the app into the image
COPY . /app

# Default streamlit port
EXPOSE 8501

# Run the app
ENTRYPOINT ["streamlit", "run", "src/Home.py", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]
