from flask import Flask
from flask_restful import Api
from dotenv import load_dotenv

from api.message_handler import Message

load_dotenv()

app = Flask(__name__)
api = Api(app)

api.add_resource(Message, '/message')

# Route mặc định để kiểm tra hoạt động
@app.route('/')
def index():
    return "Flask chatbot is running!"

if __name__ == '__main__':
    app.run(debug=True)