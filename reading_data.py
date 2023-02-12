""" Data Reading """

# Read Dataset
read_data = pd.read_csv("/content/train.csv")
print(read_data.shape)

data = read_data.drop(['id', 'keyword', 'location'], axis=1)
data.shape

data.isnull().sum()
data = data.drop_duplicates()

label_data = data.drop(['text'], axis=1)
train_data = data['text']
print(train_data.shape)
print(label_data.shape)