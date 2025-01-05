import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Tải dữ liệu từ tệp intents.json
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Khởi tạo lemmatizer từ NLTK để xử lý từ
lemmatizer = WordNetLemmatizer()

# Tiền xử lý dữ liệu
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Lưu danh sách từ và nhãn vào tệp để dùng lại
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Tạo dữ liệu huấn luyện
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Chuyển đổi dữ liệu huấn luyện thành numpy array
random.shuffle(training)
training = np.array(training)

# Tách dữ liệu thành train và test (80% train, 20% test)
train_x, test_x, train_y, test_y = train_test_split(training[:, :len(words)], training[:, len(words):], test_size=0.2, random_state=42)

# Xây dựng mô hình với Hyperparameter Tuning
model = tf.keras.Sequential()

# Thêm lớp Dense và điều chỉnh số lượng neurons
model.add(tf.keras.layers.Dense(256, input_shape=(len(train_x[0]),), activation='relu'))  # Thêm số neurons
model.add(tf.keras.layers.Dropout(0.5))  # Điều chỉnh tỷ lệ dropout
model.add(tf.keras.layers.Dense(128, activation='relu'))  # Lớp Dense tiếp theo
model.add(tf.keras.layers.Dropout(0.5))  # Điều chỉnh tỷ lệ dropout

# Lớp Output
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Biên dịch mô hình với Optimizer Adam và điều chỉnh Learning Rate
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Điều chỉnh learning rate
              metrics=['accuracy'])

# Huấn luyện mô hình với dữ liệu huấn luyện
model.fit(train_x, train_y, epochs=200, batch_size=32, verbose=1)

loss, accuracy = model.evaluate(test_x, test_y)
print(f"Độ chính xác của mô hình trên tập kiểm tra: {accuracy * 100:.2f}%")

model.save('chatbot_model.h5')
print("Mô hình huấn luyện xong và đã được lưu!")

