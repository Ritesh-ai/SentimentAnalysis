from sentiment import TrainingAndTesting 
'''
below used variables and info
---------
traning_data: your traning data (list or tuple of document)
traning_label: label of corresponding traning_data (list or tuple of lables)
testing_data: your traning data (list or tuple of document)
test_label: label of testing_data (list or tuple)
'''

import pandas as pd

df = pd.read_excel("reviews.xlsx")
X = df['sentence'].tolist()
y = df['rating'].tolist()
# training_data = df['sentence'].tolist()
# training_label = df['rating'].tolist()

# with open("data.txt","r") as f:
#     data = f.read()
# testing_data = []
# testing_label = []
# for i in data.split("\n")[:-1]:
#     testing_data.append(i.split("\t")[0])
#     testing_label.append(i.split("\t")[1])



# print(len(testing_data))

from sklearn.model_selection import train_test_split
training_data, testing_data, training_label, testing_label = train_test_split(X, y, test_size=0.2, random_state=42)

# print(training_label[:10])

train_and_test_object = TrainingAndTesting()
train_and_test_object.train_me(training_data, training_label)
# test_result = train_and_test_object.test_me(testing_data)
# print(TrainingAndTesting.model_evaluation(test_result, testing_label))



from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

while True:
    text = input("Enter : ")
    # print(text)
    test_result = train_and_test_object.test_me(text)
    print(test_result)
    score = analyser.polarity_scores(text)
    print(score)
    if text == 'no':
        break
