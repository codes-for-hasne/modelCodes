import json

from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath

# leaves and total nodes
num = 1840
total = 8801



file_path = datapath('path of the original dataset for poincare')
model = PoincareModel(PoincareRelations(file_path,delimiter=','), negative=2, size=32)
model.train(epochs=10000,print_every=10)

child2ParentDict = {}

for each in range(total):
    r = model.kv.closest_parent(str(each))
    if r is None:
        print(each)
    else:
        child2ParentDict[each] = r


parent2ChildDict = {}
for child in child2ParentDict:
    parent = child2ParentDict[child]
    if(parent in parent2ChildDict):
        parent2ChildDict[parent].append(child)
    else:
        parent2ChildDict[parent] = [child]


text = json.dumps(parent2ChildDict)
f = open('path to a tmp collection file','w')
f.writelines(text)
f.close()
