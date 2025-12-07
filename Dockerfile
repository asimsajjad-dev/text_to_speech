# Use a Python version compatible with Coqui TTS (VITS)
FROM python:3.10-slim

# Install system dependencies, including FFmpeg (essential for pydub)
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .
# Use a non-upgrade install to avoid conflicts
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Command to run the Gradio application
CMD ["python", "app.py"]
