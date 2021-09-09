import pymongo as pym
import pprint

client = pym.MongoClient('mongodb://localhost:27017/')
db = client.Vieweet_prototype_01
models = db.models

#tests
pprint.pprint(models.find_one())
pprint.pprint(models.find_one({"classifier":"int_ext_5"}))

for model in models.find():
    pprint.pprint(model)

test_count = models.find({"accuracy": [{"$gte": 0.90}]})
test_count.count_documents()

#retrieve relevant models
classifier_compare = "int_ext"
field_compare = "accuracy"

classifiers = models.find({classifier_compare: 1, field_compare: 1})

#classifiers to classifier_list

int_ext_all = []
for classifier in classifiers: #classifier_list:
    if classifier_compare in classifier:
        int_ext_all += classifier_compare

pprint(int_ext_all)
