# from flask import request, jsonify
# from flask_restful import Resource, reqparse
# from services.open_ai_service import get_openai_response

# parser = reqparse.RequestParser()
# parser.add_argument('message')

# class Message(Resource):

#     def post(self):
#         args = parser.parse_args()
#         try:
#             if not args['message']:
#                 return jsonify({"error": "No message provided"}), 400
            
#             bot_reply = get_openai_response(args['message'])
            
#             return jsonify({"reply": bot_reply})
#         except Exception as e:

#             return jsonify({"error": str(e)}), 500



from flask import request, jsonify
from flask_restful import Resource, reqparse
from services.open_ai_service import get_openai_response
from services.open_ai_service import get_response  # Import get_response
from services.open_ai_service import predict_tag

# Tạo parser để phân tích yêu cầu
parser = reqparse.RequestParser()
parser.add_argument('message', type=str, required=True, help="Message from user is required")

class Message(Resource):
    def post(self):
        args = parser.parse_args()  # Phân tích tham số yêu cầu
        try:
            user_message = args['message']  # Lấy thông điệp từ người dùng
            
            if not user_message:
                return jsonify({"error": "No message provided"}), 400
            
            # Dự đoán tag với TensorFlow
            predicted_tag = predict_tag(user_message)  # Gọi hàm dự đoán tag
            
            if predicted_tag:
                # Nếu có tag được dự đoán, lấy phản hồi từ intents
                bot_reply = get_response(predicted_tag)  # Bạn cần định nghĩa hàm get_response
            else:
                # Nếu không có tag, gọi OpenAI API để lấy phản hồi
                bot_reply = get_openai_response(user_message)
            
            return jsonify({"reply": bot_reply})  # Trả về phản hồi dưới dạng JSON
        except Exception as e:
            return jsonify({"error": str(e)}), 500  # Trả về lỗi nếu có ngoại lệ xảy ra
