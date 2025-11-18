# Housing Price Prediction API (MLOps Project)

This project takes a trained XGBoost model (from the previous data science project) and deploys it as a production-ready REST API using FastAPI and Docker.

The API receives house features as a JSON request, predicts the sale price using the loaded model, and returns the prediction.

## Key Technologies

FastAPI: For building the high-performance API.

Docker: For containerizing the application, ensuring consistency across environments.

Scikit-learn/XGBoost: For serving the pre-trained model pipeline.

Uvicorn: As the ASGI server to run the API.

## Project Structure

housing_price_api/
├── app/
│   ├── __init__.py
│   ├── main.py             (The FastAPI application logic)
│   └── model/
│       └── best_model.pkl  (The pre-trained model pipeline copied from Project 4)
|       └── feature_list.pkl
├── requirements.txt
├── Dockerfile              (Instructions for building the Docker image)
└── README.md


## How to Run (Locally, without Docker)

### Create Environment:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


### Run Server:
From the housing_price_api/ root directory, run:

uvicorn app.main:app --reload


### Test API:
Open your browser and go to http://127.0.0.1:8000/docs. You will see the automatic Swagger UI documentation, where you can test the /predict endpoint interactively.

## How to Run (with Docker)

This is the recommended way to run the application in a production-like environment.

### Build the Image:

docker build -t housing_api .


### Run the Container:
This command runs the container in the background (-d) and maps port 8000 of your machine to port 8000 of the container.

docker run -d -p 8000:8000 --name housing_api_container housing_api


### Test API:
Open http://127.0.0.1:8000/docs in your browser. The API is now running inside the container!

### Stop & Remove Container:

docker stop housing_api_container
docker rm housing_api_container
