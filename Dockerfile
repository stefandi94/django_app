# The first instruction is what image we want to base our container on
# We Use an official Python runtime as a parent image
FROM python:3.6

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1

# create root directory for our project in the container
RUN mkdir /app

# Set the working directory to /music_service
WORKDIR /app

# Copy the current directory contents into the container at /music_service
ADD ./ /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r ml/requirements.txt

# Migrates the database, uploads staticfiles, and runs the production server
RUN ls -la ./scripts
#RUN chmod -R o+x scripts/start.sh

EXPOSE 8000
ENTRYPOINT ["bash", "chmod -R o+x scripts/start.sh"]
RUN ls -la ./scripts
ENTRYPOINT ["scripts/start.sh"]
