""" Long Short-Term Memory"""

# Tokenize the texts
max_features=3000
tokenizer=Tokenizer(num_words=max_features,split=' ')
tokenizer.fit_on_texts(train_data.values)
X = tokenizer.texts_to_sequences(train_data.values)
X = pad_sequences(X)      # to make the whole input text data on the same size.

tokenizer.sequences_to_texts([[ 713,  154,   56, 1434,   14]])

""" Split the Data """
X_train, X_test, y_train, y_test = train_test_split(X, label_data, test_size = 0.3, random_state =0)

max_features = 30000
embed_dim = 32

lstm_model = Sequential()
lstm_model.add(Embedding(max_features, embed_dim, input_length = X_train.shape[1]))
#lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=60, input_shape=(X_train.shape[1], 1), activation='relu', return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss = 'binary_crossentropy', optimizer='adam' , metrics = ['accuracy'])

lstm_model.fit(X_train, y_train, epochs = 10, batch_size=1, validation_data=(X_test, y_test))

y_pred = lstm_model.predict(X_test).round()
# Final evaluation of the model
scores = lstm_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

train_accuracy = round(metrics.accuracy_score(y_train, lstm_model.predict(X_train).round())*100)
train_accuracy

accuracy = round(accuracy_score(y_test,y_pred),3)
precision = round(precision_score(y_test,y_pred,average='weighted'),3)
recall = round(recall_score(y_test,y_pred,average='weighted'),3)

print(f'Accuracy of the model: {np.round(accuracy*100,2)}%')
print(f'Precision Score of the model: {np.round(precision*100,2)}%')
print(f'Recall Score of the model: {np.round(recall*100,2)}%')
print('-'*50)
print(classification_report(y_test,y_pred))