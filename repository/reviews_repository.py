import pandas as pd
import json 
import asyncio

data = None

async def get_json_data():
    data = None
    with open('repository/reviews.json') as f:
        data = json.load(f)
    return data