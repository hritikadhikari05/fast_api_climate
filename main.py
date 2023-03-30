from fastapi import FastAPI,File,UploadFile
from  foodPredModel import FoodPredction
import numpy as np
import os
import uvicorn
import pickle
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

xg_boost_model = str(os.getcwd()) +"\\"+ "xgboostalgo_food_chain.pkl"
decison_model = str(os.getcwd()) +"\\"+ "decisiontreealgo_food_chain.pkl"
print("THis is path",decison_model)

# set up allowed origins
origins = [
    "http://localhost",
    "http://127.0.0.1:8000"
    "http://localhost:8080",
    "http://localhost:3000",
    "https://example.com",
    "https://www.example.com",
]

# add CORS middleware to your application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pickle_in = open(xg_boost_model, "rb")
classifier = pickle.load(pickle_in)

pickle_decision_tree = open(decison_model, "rb")
classifier_decision_tree = pickle.load(pickle_decision_tree)

@app.get("/")
async def index():
    return {"message": "Hello World"}

@app.post("/predict-xgboost")
async def predict(data: FoodPredction):
    data = data.dict()
    avg_rainfall = data['avg_rainfall']
    pesticides = data['pesticides']
    avg_temp = data['avg_temp']
    item = data['item']
    prediction = classifier.predict([[avg_rainfall, pesticides, avg_temp, item]])
    
    return {
        "prediction": str(prediction[0])
    }


@app.post("/predict-decision-tree/")
async def predict_decision_tree(file: UploadFile = File(...)):
    # Save the uploaded file
    with open(file.filename, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # Read the uploaded file
    df_yield_pred = pd.read_csv(file.filename)
    # Convert into numpy array
    Z = df_yield_pred.iloc[:, 1:].values 
    # Predict the yield
    prediction = classifier_decision_tree.predict(Z)
    # Condition to check if the prediction is None
    if(prediction is None):
        return {"message": "No prediction"}
    else:
        # set the file path
        file_path = file.filename
        # check if the file exists
        if os.path.exists(file_path):
            # delete the file
            os.remove(file_path)
            print(f"File {file_path} deleted.")
        else:
            print(f"File {file_path} does not exist.")
        return {"message": str(prediction[0])}


    
    # print(Z.shape)
    # Return a success message

if __name__ == "__main__":
    # Create a directory for storing the uploaded files
    UPLOAD_DIR = "uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
