
import fastapi
import pandas as pd
from fastapi import HTTPException, Depends
from pydantic import BaseModel
from model import DelayModel

app = fastapi.FastAPI()
delay_model = DelayModel()

class PredictRequest(BaseModel):
    data: list[dict]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    try:
        # Convert data to dataframe
        data = pd.DataFrame(request.data)
        
        # Preprocess data
        features = delay_model.preprocess(data)
        
        # Make predictions
        predictions = delay_model.predict(features)
        
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

