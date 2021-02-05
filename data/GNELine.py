import json

import torch

import codes.utils.data_etl as etl

# number of nodes
nums = 4000

treePath = "path to the tree file"
tree, total_level, all_leaves = etl.prepare_tree(treePath)



f = open("path to the results of GNE", encoding='utf-8')
content = f.read()
dict = json.loads(content)
embeddings = dict['coordinates']
radii = dict['radius']
len = len(radii)

# We draw 10 layers
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
for layer in layerDict:

    if layer == 10:
        continue

    nodeList = layerDict[layer]
    for eachNode in nodeList:
        curTensor = torch.Tensor(embeddings[eachNode])
        nodeDist = torch.norm(curTensor - allTensor, dim=1)
        radius = radii[eachNode]
        tmp = nodeDist - torch.Tensor([radius])
        tmp = torch.abs(tmp)
        tmp = torch.where(tmp <= 1, torch.ones(len), torch.zeros(len))
        index = torch.nonzero(tmp)
        children = torch.squeeze(index).numpy().tolist()
        if isinstance(children, list):
            parent2ChildDict[eachNode] = children
        else:
            parent2ChildDict[eachNode] = [children]

text = json.dumps(parent2ChildDict)
f = open('path to a tmp collection file','w')
f.writelines(text)
f.close()