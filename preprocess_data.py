""" Clean Data """

def lower(text):
  return str(text.lower())

def remove_numbers(text):
  return re.sub('\d+', '', text)

def remove_html_tags(text):
  return re.sub('\[.*?\]', '', text)

def remove_url(text):
  return re.sub('https?://\S+|www\.\S+', '', text)

def remove_punctuations(text):
  return re.sub('[%s]' % re.escape(string.punctuation),'',text)

def remove_stop_words(text):
  return ' '.join([word for word in text if word.lower() not in stop_words])

def lemmatize_data(text):
  text = ' '.join(stemm.stem(word) for word in text.split(' '))
  text = ' '.join(lemma.lemmatize(word) for word in text.split(' '))
  return text

def wordTokenize(text):
  text = tokenize.tokenize(text)
  return text
  
""" Call Functions """
def preprocess(text):
  text = lower(text)
  text = remove_numbers(text)
  text = remove_html_tags(text)
  text = remove_url(text)
  text = remove_punctuations(text)
  text = wordTokenize(text)
  text = remove_stop_words(text)
  text = lemmatize_data(text)
  return text

train_data = train_data.apply(preprocess)
train_data