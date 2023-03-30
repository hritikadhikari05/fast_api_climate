from pydantic import BaseModel


class FoodPredction(BaseModel):
    avg_rainfall: float
    pesticides: float
    avg_temp: float
    item: float
    