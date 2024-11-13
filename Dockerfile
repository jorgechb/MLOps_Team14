FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the model and server code
COPY models/model.pkl /app/
COPY src/api_test.py /app/
COPY data/processed/transformed/ /app/data/processed/transformed/

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn pydantic pandas

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api_test:app", "--host", "0.0.0.0", "--port", "8000"]
