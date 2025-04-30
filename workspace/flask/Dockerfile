# Use official Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy Flask app files into the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# ENV PYTHONPATH="usr/irissys/lib/python3.10/site-packages:/opt/conda/lib/python3.10/site-packages:/app"

# Expose the Flask app port
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py", "--debug", "true"]
# CMD ["sh", "-c", "python iris_db.py && python app.py"]

