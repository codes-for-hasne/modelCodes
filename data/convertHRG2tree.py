import re

if __name__ == "__main__":


    leavesDict = {}
    innerDict = {}
    n=4000
    layers = 10



    f = open("path to the results of HRG", "r")
    for line in f.readlines():
        line = line.replace(' ','').strip()
        innerNode = re.findall(r'\[(.*)\]', line)
        innerNode = innerNode[0]
        leftNode = re.findall(r'L=(.*)R',line)
        leftNode = leftNode[0]

        rightNode = re.findall(r'R=(.*)p=', line)
        rightNode = rightNode[0]


        if(leftNode.find("(D)") > 0):
            leftNode = leftNode.replace('(D)', '').strip()
            innerDict[leftNode] = innerNode
        else:
            leftNode = leftNode.replace('(G)','').strip()
            leavesDict[leftNode] = innerNode

        if (rightNode.find("(D)") > 0):
            rightNode = rightNode.replace('(D)', '').strip()
            innerDict[rightNode] = innerNode
        else:
            rightNode = rightNode.replace('(G)', '').strip()
            leavesDict[rightNode] = innerNode
    f.close()


    res = {}
    for i in range(n):
        i = str(i)
        res[i] = [i]
        parent = leavesDict[i]
        res[i].append(parent)
        while (parent in innerDict):
            res[i].append(innerDict[parent])
            parent = innerDict[parent]

    maxDepth = 0
    for k in res:
        maxDepth = len(res[k]) if len(res[k]) > maxDepth else maxDepth

    innerNodePadding = n


    for k in res:
        while(len(res[k])< maxDepth):
            res[k].insert(1, innerNodePadding)
            innerNodePadding+=1


    for k in res:
        for i in range(len(res[k])):
            if (i > 0):
                res[k][i] = n + int(res[k][i])
            else:
                res[k][i] = int(k)


    layers = layers
    layersArr = [0 for i in range(layers)]
    layerGap = int(maxDepth / (layers-1))

    for i in range(len(layersArr)):
        times = i
        if(i==len(layersArr) - 1):
            layersArr[i] = maxDepth - 1
        else:
            layersArr[i] = times * layerGap


    f = open("path to file to store the tree", "w+")

    # re-number the nodes
    innerNodeNewIndex = {}
    cnt = n
    strSet = set()

    for i in range(len(layersArr) - 1):
        for k in res:
            tmp = res[k]
            v1 = tmp[layersArr[i]]
            v2 = tmp[layersArr[i + 1]]

            if (v1 > (n - 1)):
                if v1 in innerNodeNewIndex:
                    v1 = innerNodeNewIndex[v1]
                else:
                    innerNodeNewIndex[v1] = cnt
                    v1 = cnt
                    cnt += 1
            if (v2 > (n - 1)):
                if v2 in innerNodeNewIndex:
                    v2 = innerNodeNewIndex[v2]
                else:
                    innerNodeNewIndex[v2] = cnt
                    v2 = cnt
                    cnt += 1

            str1 = str(v2) + "\t" + str(v1) + "\r\n"
            if( str1 not in strSet ):
                strSet.add(str1)
                f.write(str1)
    f.close()







