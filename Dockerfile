# base image
FROM python:3
# Copy application code
WORKDIR /var/sentiment/
COPY . .
COPY requirements.txt .
# Fetch app specific dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Expose port
EXPOSE 5000
# Start the app
WORKDIR /var/sentiment/mycode/
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:5000"]