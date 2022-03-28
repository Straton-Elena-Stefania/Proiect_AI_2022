import gensim as gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re, string
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from string import punctuation, digits
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Activation, GRU
import tensorflow as tf
EMBEDDING_DIM = 200
col_list = ["tweet", "sarcastic", "sarcasm", "irony", "satire", "understatement", "overstatement",
            "rhetorical_question"]
dataset = pd.read_csv("data.csv", usecols=col_list)

stop = set(stopwords.words('english'))
punct = list(string.punctuation)
stop.update(punct)


def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


def delete_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)  # no emoji


def delete_digits(text):
    text = text.lower()
    clean = text.translate(str.maketrans('', '', digits))
    return clean


def delete_punctuation(text):
    clean = text.translate(str.maketrans('', '', punctuation + '’“”'))
    return clean


# Removing the noisy text
def denoise_text(text):
    text = delete_punctuation(text)
    text = remove_stopwords(text)
    text = delete_emoji(text)
    text = delete_digits(text)
    return text


def lemm(text):
    lemmatizer = WordNetLemmatizer()
    sar_list_lemmatizer = [lemmatizer.lemmatize(word) for word in text]
    return sar_list_lemmatizer


def get_weight_matrix(model, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in vocab.items():
        weight_matrix[i] = model.__getitem__(word)
    return weight_matrix


def make_word_model(wordset):
    w2v_model = gensim.models.Word2Vec(sentences=wordset, vector_size=EMBEDDING_DIM, window=5, min_count=1)

    tokenizer = text.Tokenizer(num_words=36173)
    tokenizer.fit_on_texts(wordset)
    tokenized_train = tokenizer.texts_to_sequences(wordset)
    x = sequence.pad_sequences(tokenized_train, maxlen=20)
    vocab_size = len(tokenizer.word_index) + 1
    embedding_vectors = get_weight_matrix(w2v_model.wv, tokenizer.word_index)

    return x, vocab_size, embedding_vectors


dataset['tweet'] = dataset['tweet'].apply(denoise_text)
sarcastic_list = dataset['tweet'][:867]
nosarcastic_list = dataset['tweet'][867:]




from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn import metrics
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',ngram_range=(1,1), tokenizer=token.tokenize)
text_counts = cv.fit_transform(dataset['tweet'])

x_train,x_test,y_train,y_test = train_test_split(text_counts, dataset['sarcastic'], test_size=0.25, random_state=5)
#MNB = MultinomialNB()
MNB = BernoulliNB()
MNB.fit(x_train,y_train)
predicted = MNB.predict(x_test)
accuracy = metrics.accuracy_score(predicted, y_test)
print("Accuracy for Bernoulli Naive Bayes")
print(str('{:04.2f}'.format(accuracy*100)))

dataset['tweet'] = dataset['tweet'].apply(word_tokenize)
dataset['tweet'] = dataset['tweet'].apply(lemm)

result = []
for l in range(0, 867):
    result.append([dataset["sarcasm"][l], dataset["irony"][l], dataset["satire"][l], dataset["understatement"][l]
                      , dataset["overstatement"][l], dataset["rhetorical_question"][l]])
result = np.array(result)



x, vocab_size, embedding_vectors = make_word_model(dataset['tweet'])

model = Sequential()
model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=20, trainable=True))
model.add(Dense(256))
model.add(Bidirectional(LSTM(96, dropout=0.2, recurrent_dropout=0.2, activation='relu')))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

del embedding_vectors

# model.summary()

x_train, x_test, y_train, y_test = train_test_split(x, dataset['sarcastic'], test_size=0.4, random_state=0)

history = model.fit(x_train, y_train, batch_size=128, validation_data=(x_test, y_test), epochs=3)

print("Accuracy of the model on Training Data is - ", model.evaluate(x_train, y_train)[1] * 100)
print("Accuracy of the model on Testing Data is - ", model.evaluate(x_test, y_test)[1] * 100)

x, vocab_size, embedding_vectors = make_word_model(sarcastic_list)

model = Sequential()
model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=20, trainable=True))
model.add(Dense(256))
model.add(Dense(128))
model.add(Bidirectional(GRU(32)))
model.add(Dense(6, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['acc'])

del embedding_vectors

# model.summary()

x_train, x_test, y_train, y_test = train_test_split(x, result, test_size=0.2, random_state=0)

history2 = model.fit(x_train, y_train, batch_size=128, validation_data=(x_test, y_test), epochs=10)

print("Accuracy of the model on Training Data is - ", model.evaluate(x_train, y_train)[1] * 100)
print("Accuracy of the model on Testing Data is - ", model.evaluate(x_test, y_test)[1] * 100)

epochs = [i for i in range(3)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['acc']
train_acc2 = history2.history['acc']
val_acc = history.history['val_acc']
val_acc2 = history2.history['val_acc']
fig.set_size_inches(20, 10)

ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label='Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

epochs2 = [i for i in range(10)]
ax[1].plot(epochs2, train_acc2, 'go-', label='Training Accuracy')
ax[1].plot(epochs2, val_acc2, 'ro-', label='Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")

plt.show()
