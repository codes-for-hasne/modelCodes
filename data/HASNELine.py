import json

import torch

import codes.utils.data_etl as etl

nums = 4000

treePath = "path to the tree file"
tree, total_level, all_leaves = etl.prepare_tree(treePath)

f = open("path to results of HASNE", encoding='utf-8')
content = f.read()
dict = json.loads(content)
embeddings = dict['embedding']
range = dict['circleRange']

len = len(embeddings)

# We dran 10 layers
layerDict = {
    1 : [],
    2 : [],
    3 : [],
    4 : [],
    5 : [],
    6 : [],
    7 : [],
    8 : [],
    9 : [],
    10 : []
}

for node in tree:
    level = node.level
    if (level > 0 and level < 11):
        layerDict[level].append(node.id)
    else:
        print("error in tree:"+str(node.id))

allTensor = torch.Tensor(embeddings)
parent2ChildDict = {}
child2ParentDict = {}



allLower, allUpper = torch.chunk(allTensor, 2, dim=1)


for layer in layerDict:

    if layer == 1:
        continue

    nodeList = layerDict[layer]
    for eachNode in nodeList:
        curTensor = torch.Tensor(embeddings[eachNode])

        parentTensorInfered = curTensor - torch.Tensor([range])
        parentTensorInferedLower, parentTensorInferedUpper = torch.chunk(parentTensorInfered, 2, dim=0)

        maxLower = torch.where( parentTensorInferedLower > allLower, parentTensorInferedLower, allLower )
        minUpper = torch.where( parentTensorInferedUpper < allUpper, parentTensorInferedUpper, allUpper )
        overlap = (minUpper - maxLower).sum(dim=1)

        child2ParentDict[eachNode] = torch.max(overlap,0)[1].item()


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