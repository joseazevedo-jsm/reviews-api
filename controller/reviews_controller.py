# reviews_controller.py
from flask import Blueprint, jsonify
from service.reviews_service import get_data
import json
import asyncio

reviews_blueprint = Blueprint('reviews', __name__)

@reviews_blueprint.route('/')
async def get_json():
    data = await get_data()
    return jsonify(data)