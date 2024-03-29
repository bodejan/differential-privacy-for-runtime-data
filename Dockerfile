# Base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Set the working directory to the app directory
WORKDIR /app/dashboard

# Expose the port on which the Flask app will run
EXPOSE 8050

# Set the command to run the Flask app
CMD ["python", "app.py"]
