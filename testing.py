# from sentiment import TrainingAndTesting 
# '''
# below used variables and info
# ---------
# query: get reviews related to this query eg. computer engineering books
# total_reviews: total no of reviews you want to extract
# reviews_rating: reviews_rating[0] is list conatin reviews and reviews_rating[1] is list contain rating of corresponding reviews in  reviews_rating[0]
# '''
# obj = TrainingAndTesting()
# reviews_rating = obj.get_amazon_data( query='python books', total_reviews=100)

# print(reviews_rating)


from sentiment import TrainingAndTesting 
'''
below used variables and info
---------
traning_data: your traning data (list or tuple of document)
traning_label: label of corresponding traning_data (list or tuple of lables)
testing_data: your traning data (list or tuple of document)
test_label: label of testing_data (list or tuple)
'''

with open("data.txt","r") as f:
    data = f.read()
X = []
y = []
for i in data.split("\n")[:-1]:
    X.append(i.split("\t")[0])
    y.append(i.split("\t")[1])

# print(X[0],"-----",y[0])


from sklearn.model_selection import train_test_split
training_data, testing_data, training_label, testing_label = train_test_split(X, y, test_size=0.2, random_state=42)



train_and_test_object = TrainingAndTesting()
train_and_test_object.train_me(training_data, training_label)
# test_result = train_and_test_object.test_me(testing_data)
# print(TrainingAndTesting.model_evaluation(test_result, testing_label))

while True:
    text = input("Enter : ")
    # print(text)
    test_result = train_and_test_object.test_me(text)
    print(test_result)
    # score = analyser.polarity_scores(text)
    # print(score)
    if text == 'no':
        break