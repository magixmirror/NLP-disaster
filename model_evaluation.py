""" Evaluation the model """

test_data = pd.read_csv('test.csv')
test_data.head()

data = test_data.drop(['keyword', 'location'], axis=1)
data.shape
id = data['id']
id = id.to_frame()
id = pd.DataFrame(id)
id

data.isnull().sum()
data = data.drop_duplicates()

test_data = test_data['text'].apply(preprocess)
test_data