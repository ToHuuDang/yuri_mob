from flask import request, jsonify
from flask_restful import Resource, reqparse
from services.open_ai_service import get_openai_response

parser = reqparse.RequestParser()
parser.add_argument('message')

class Message(Resource):

    def post(self):
        args = parser.parse_args()
        try:
            if not args['message']:
                return jsonify({"error": "No message provided"}), 400
            
            bot_reply = get_openai_response(args['message'])
            
            return jsonify({"reply": bot_reply})
        except Exception as e:

            return jsonify({"error": str(e)}), 500
