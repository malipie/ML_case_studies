# --- Stage 1: Base Image ---
# Start from an official, lightweight Python 3.12 image
FROM python:3.12-slim

# --- Stage 2: Set Up Environment ---
# Set the working directory inside the container
WORKDIR /app

# Set environment variables for optimal Python performance
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Stage 3: Install Dependencies ---
# First, copy *only* the requirements.txt file
# This is an optimization trick: Docker won't reinstall packages
# every time you change your code, only when requirements.txt changes.
COPY requirements.txt .

# Run the pip install
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy Application Files ---
# Now, copy the rest of our application (the 'app' folder)
COPY ./app /app/app

# --- Stage 5: Run the Application ---
# Tell Docker that our application will run on port 8000
EXPOSE 8000

# The command that will run when the container starts
# We use uvicorn and host on '0.0.0.0' to make it accessible
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]