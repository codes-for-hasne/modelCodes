import json

import codes.utils.data_etl as etl

target = 4000
treePath = "path to the tree file"
tree, total_level, all_leaves = etl.prepare_tree(treePath)

# We display 10 layers
layerDictGroundTruth = {
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

nodeDictGroundTruth = {}

for node in tree:
    nodeDictGroundTruth[node.id] = node.direct_children
    level = node.level
    if (level > 0 and level < 11):
        layerDictGroundTruth[level].append(node.id)
    else:
        print("error in tree:"+str(node.id))

baseDir = "data/"
# hasne tmp collection
hasnePath = baseDir + "path to the tmp collection file"
# gne tmp collection
gnePath = baseDir + "path to the tmp collection file"
# poincare tmp collection
poincarePath = baseDir + "path to the tmp collection file"

fhasne = open(hasnePath, encoding='utf-8')
contentHasne = fhasne.read()
dictHasne = json.loads(contentHasne)
fhasne.close()

fgne = open(gnePath, encoding='utf-8')
contentGne = fgne.read()
dictGne = json.loads(contentGne)
fgne.close()

fpoincare = open(poincarePath, encoding='utf-8')
contentPoincare = fpoincare.read()
dictPoincare = json.loads(contentPoincare)
fpoincare.close()


totalLayerGne = {
    1:0,
    2:0,
    3:0,
    4:0,
    5:0,
    6:0,
    7:0,
    8:0,
    9:0
}
totalLayerHasne = {
    1:0,
    2:0,
    3:0,
    4:0,
    5:0,
    6:0,
    7:0,
    8:0,
    9:0
}
totalLayerPoincare = {
    1:0,
    2:0,
    3:0,
    4:0,
    5:0,
    6:0,
    7:0,
    8:0,
    9:0
}


def jaccardSim(a,b):
    """
    calculate the jaccard sim
    :param a:
    :param b:
    :return:
    """
    unions = len(set(a).union(set(b)))
    intersections = len(set(a).intersection(set(b)))
    return 1. * intersections / unions

for layer in layerDictGroundTruth:
    # pass
    if layer == 10:
        continue


    branchNodes = layerDictGroundTruth[layer]
    cnt = len(branchNodes)
    for each in branchNodes:

        groundTruthSet = set(nodeDictGroundTruth[each])
        each = str(each)
        if(each in dictGne):
            gneSet = set(dictGne[each])
        else:
            gneSet = set()
        if(each in dictHasne):
            hasneSet = set(dictHasne[each])
        else:
            hasneSet = set()

        if(each in dictPoincare):
            poincareSet = set(dictPoincare[each])
        else:
            poincareSet = set()

        jaccardGne = jaccardSim(groundTruthSet, gneSet)
        jaccardHasne = jaccardSim(groundTruthSet,hasneSet)
        jaccardPoincare = jaccardSim(groundTruthSet, poincareSet)
        totalLayerGne[layer] += jaccardGne
        totalLayerHasne[layer] += jaccardHasne
        totalLayerPoincare[layer] += jaccardPoincare


    print("GNE_layer_"+str(layer)+":"+str(totalLayerGne[layer] / cnt))
    print("HASNE_layer_" + str(layer) + ":" + str(totalLayerHasne[layer] / cnt))
    print("Poincare_layer_" + str(layer) + ":" + str(totalLayerPoincare[layer] / cnt))
    print("\r\n")
