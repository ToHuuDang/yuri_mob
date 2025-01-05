# import os
# from openai import OpenAI
# from data_service.data_service import get_product_data
# from flask import jsonify
# import json

# def render_context():
#     products = get_product_data()

#     product_context = "Danh sách phòng hiện có:\n"
#     for product in products:
#         product_context += f"\n- {product['id']}: {product['address']}, Giá internet: {product['internet_cost']}, Mô tả phòng: {product['description']}, Tên phòng: {product['title']}\n"

#     return product_context

        
# def get_openai_response(message):
#     client = OpenAI(
#         api_key=os.getenv("OPENAI_API_KEY")
#     )

#     context = render_context()

#     messages = [
#          {
#             "role": "system", 
#             "content": f""" 
#                 Bạn là một tư vấn viên chuyên nghiệp về phòng trọ.
#                 Hãy sử dụng thông tin của các phòng trọ được cung cấp để tư vấn cho khách hàng.
#                 Những thông tin bạn được cung cấp là:{context}. 
#                 Đưa ra những đề xuất phù hợp dựa trên câu hỏi của họ."""
#         },
#         {"role": "user", "content": f"Câu hỏi của khách hàng: {message}"}
#     ]
    
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages
#         )
        
#         return response.choices[0].message.content
        
#     except Exception as e:
#         print(f"Error while communicating with OpenAI: {str(e)}")
#         return None

# /**********************************************************************************************/
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import os
from openai import OpenAI
from nltk.stem import WordNetLemmatizer
import nltk
from flask_restful import Resource, reqparse
from data_service.data_service import get_product_data

# Tải mô hình và dữ liệu
lemmatizer = WordNetLemmatizer()
model = tf.keras.models.load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Hàm tiền xử lý câu
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Tạo bag-of-words dữ liệu thành vector
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Dự đoán tag
def predict_tag(sentence):
    bow_input = bow(sentence, words)
    res = model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.7
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else None

# Lấy câu trả lời từ intents
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return None

# Render context từ dữ liệu phòng trọ
def render_context():
    products = get_product_data()

    product_context = "Danh sách phòng hiện có:\n"
    for product in products:
        product_context += f"\n- {product['id']}: {product['address']}, Giá internet: {product['internet_cost']}, Mô tả phòng: {product['description']}, Tên phòng: {product['title']}, Giá phòng: {product['price']}\n"

    return product_context

# Gọi OpenAI API
def get_openai_response(message):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = render_context()

    messages = [
        {
            "role": "system", 
            "content": f""" 
                Bạn là một tư vấn viên chuyên nghiệp về phòng trọ.
                Hãy sử dụng thông tin của các phòng trọ được cung cấp để tư vấn cho khách hàng.
                Những thông tin bạn được cung cấp là:{context}. 
                Đưa ra những đề xuất phù hợp dựa trên câu hỏi của họ.
                Nếu người dùng hỏi số lượng phòng trống thì chỉ cần đưa ra tên và số lượng phòng còn trống"""
        },
        {"role": "user", "content": f"Câu hỏi của khách hàng: {message}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error while communicating with OpenAI: {str(e)}")
        return "Xin lỗi, hiện tại tôi không thể xử lý yêu cầu của bạn."

class Message(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('message', type=str, required=True, help="Message from user is required")
        args = parser.parse_args()
        user_message = args['message']

        # Dự đoán tag với TensorFlow
        predicted_tag = predict_tag(user_message)
        
        if predicted_tag:
            # Nếu có kết quả từ mô hình TensorFlow
            response = get_response(predicted_tag)
        else:
            # Nếu mô hình TensorFlow không thể dự đoán, sử dụng OpenAI API để trả lời
            response = get_openai_response(user_message)

        return {'response': response}, 200

