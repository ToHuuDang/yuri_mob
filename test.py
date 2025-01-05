import random
import json
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import nltk

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

# Tạo bag-of-words
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
    ERROR_THRESHOLD = 0.9
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else None

# Lấy câu trả lời từ intents
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Xin lỗi, tôi không hiểu câu hỏi của bạn."

# Chạy chatbot trên terminal
print("Chatbot đã sẵn sàng! (Nhập 'thoát' để kết thúc)")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() == "thoát":
        print("Chatbot: Tạm biệt! Hẹn gặp lại.")
        break

    # Dự đoán và phản hồi
    predicted_tag = predict_tag(user_input)
    if predicted_tag:
            response = get_response(predicted_tag)
            response_type = type(response)  # Lấy kiểu dữ liệu của phản hồi
            print(f"Bot: {response} (Kiểu dữ liệu: {response_type})")  # In ra câu trả lời và kiểu dữ liệu
    else:
            print("Bot: Xin lỗi, tôi không hiểu câu hỏi của bạn.")
    
    print(f"Chatbot: {response}")
