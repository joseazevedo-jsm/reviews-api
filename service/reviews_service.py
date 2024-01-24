from util._process_sentiment import process_sentiment
from util._process_embeddings import process_embeddings
from util._process_cluster import process_cluster
from util._visualize_cluster import visualize_cluster
from repository.reviews_repository import get_json_data
import asyncio

async def get_data():
    return await get_json_data()