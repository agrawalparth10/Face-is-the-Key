import pickle 
from sklearn.svm import SVC 
from sklearn.preprocessing import LabelEncoder


data = pickle.loads(open("./pickle/embeddings.pickle","rb").read())

le = LabelEncoder()
labels = le.fit_transform(data['names'])

classifier = SVC(C=1.0,kernel="linear",probability=True)
classifier.fit(data["embeddings"],labels)

f = open("./pickle/classifier.pickle","wb")
f.write(pickle.dumps(classifier))
f.close()

f = open("./pickle/label.pickle","wb")
f.write(pickle.dumps(le))
f.close()