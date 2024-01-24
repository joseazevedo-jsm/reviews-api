from flask import Flask,jsonify
from flask_cors import CORS
from controller.reviews_controller import reviews_blueprint

import json
 
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

app.register_blueprint(reviews_blueprint)

if __name__ == '__main__':
    app.run(debug=True)