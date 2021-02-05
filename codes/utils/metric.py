#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import os
import random
import sys
import time
import networkx as nx
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd

import torch
from matplotlib import colors
from matplotlib.patches import Ellipse, Circle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from codes.utils.filewriter import write_to_file
import codes.utils.data_etl as etl
from codes.utils.ml import MachineLearningLib as ml
import numpy as np
from sklearn.metrics import average_precision_score
import torch


class Metric(object):


    @staticmethod
    def drawAS():

        pdf_output = 'path'
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        xx = []
        yy = []
        r = 2
        stop = 3.3 * 2 * math.pi
        tmp = 0
        while tmp < stop:

            x = tmp * math.cos(tmp )
            y = tmp * math.sin(tmp )
            x = float(x)
            y = float(y)
            xx.append(x)
            yy.append(y)
            tmp = tmp + 0.001 * math.pi

        ax.plot(xx, yy, label='debug', linewidth=2)

        plt.show()

        pp = PdfPages(pdf_output)
        pp.savefig(fig)
        pp.close()






    @staticmethod
    def classification(X, params):
        """
        classification
        :param X:embeddings
        :param
        :return:
        """
        target = params['target']
        times = 10
        y = etl.get_ground_truth('path to tree file')
        # Embedding in spiral space.
        leafNodeEmbX = X[:len(y)]

        # Convert to Euclidean space.
        EmbInEuclidean = Metric.tranSpiral2Euclidean(X,'u')
        leafNodeEmbXEuclidean = EmbInEuclidean[:len(y)]
        # Which space to use: Euclidean of spiral
        use = leafNodeEmbXEuclidean

        acc = 0.0
        for _ in range(times):
            X_train, X_test, y_train, y_test = train_test_split(use, y, test_size=0.1, stratify=y)
            clf = getattr(ml, 'logistic')(X_train, y_train)
            res = ml.infer(clf, X_test, y_test)[1]
            acc += res

        acc /= float(times)
        print(acc)
        exit(1)

    @staticmethod
    def tranSpiral2Euclidean(embeddings, type='m'):
        """
        transformation into Euclidean space
        :param embeddings:
        :param type:
        :return:
        """
        embeddings = torch.tensor(embeddings)
        # find the choice of point
        singleDimNum = int(embeddings.shape[1] / 2)
        lower, higher = torch.split(embeddings, singleDimNum, dim=1)

        if(type == 'm'):
            mid = (lower + higher) * 0.5
        elif(type == 'l'):
            mid = lower
        else:
            mid = higher

        # specify the scaling factor: factor * 2 * pi = singleCircleRange
        factor = 10 * 1
        mid = mid / factor
        xs = mid.mul(torch.cos(mid))
        ys = mid.mul(torch.sin(mid))
        tar = torch.cat((xs,ys),1)

        return tar


    @staticmethod
    def visualization(embedding):
        pdf_output = 'path to the pdf result'

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(1, 1, 1)

        for index in range(len(embedding)):

            each = embedding[index]
            start = each[0]
            end = each[1]

            tmp = start
            xx = []
            yy = []
            while tmp < end:
                x = tmp * math.cos(tmp/10)
                y = tmp * math.sin(tmp/10)
                x = float(x)
                y = float(y)
                xx.append(x)
                yy.append(y)

                tmp += math.pi * 0.005
            ax.plot(xx, yy, ls='-',lw=10, label='vis')

        plt.show()

        pp = PdfPages(pdf_output)
        pp.savefig(fig)
        pp.close()

    @staticmethod
    def visualizationZoom(embedding):
        """
        visualization for a simplified version.
        :param embedding:
        :return:
        """
        pdf_output = 'path to the results'

        tree, total_level, all_leaves = etl.prepare_tree('path to tree')

        fig = plt.figure(figsize=(7, 7))

        ax = fig.add_subplot(1, 1, 1)
        # Which nodes we want to display.
        displayNode = [122, 123]

        for index in range(len(embedding)):
            mark = 0
            # root
            if index == 128:
                mark = 1

            for disp in displayNode:
                if disp in tree[index].path:
                    mark = 1
                    break

            if mark != 1:
                continue

            each = embedding[index]
            start = each[0]
            end = each[1]

            tmp = start
            xx = []
            yy = []
            while tmp < end:
                x = tmp * math.cos(tmp / 10)
                y = tmp * math.sin(tmp / 10)
                x = float(x)
                y = float(y)
                xx.append(x)
                yy.append(y)
                tmp += math.pi * 0.001

            ax.plot(xx, yy, label='vis', ls='-',lw=10)
        plt.show()

        pp = PdfPages(pdf_output)
        pp.savefig(fig)
        pp.close()



    @staticmethod
    def drawG():
        """
        visualization of the tree file
        :return:
        """

        pdf_output = 'path to result'
        fig = plt.figure(figsize=(30, 30))
        G = nx.Graph()
        i = 0
        with open('path to tree file', "r") as f:
            for line in f:
                i+=1
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                if len(items) != 2:
                    continue

                if(i%1==0):
                    G.add_edge(int(items[0]), int(items[1]))

        nx.draw(
            G,
            with_labels=False,
            node_size=1000,
            width=10,
            node_color='b',
            edge_color='r'
        )

        plt.show()

        pp = PdfPages(pdf_output)
        pp.savefig(fig)
        pp.close()

    @staticmethod
    def drawPoincare():
        """
        draw poincare vis
        :return:
        """
        model = torch.load("path to poincare results, can be train by gensim or project on github")
        embeddings = model['embeddings']
        pdf_output = 'path to the result'

        tree, total_level, all_leaves = etl.prepare_tree('path to tree file')
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(1, 1, 1)

        xx = []
        yy = []
        for index in range(len(embeddings)):

            each = embeddings[index]
            x = each[0] * 1000
            y = each[1] * 1000
            x = float(x)
            y = float(y)
            xx.append(x)
            yy.append(y)

        edges_set = set()

        for node in tree:
            path = node.path
            for i in range(len(path) - 1):
                if ((path[i], path[i + 1]) not in edges_set):
                    edges_set.add((path[i], path[i + 1]))

        for each in list(edges_set):
            xxx = []
            yyy = []
            x1 = float(embeddings[each[0]][0] * 1000)
            xxx.append(x1)
            y1 = float(embeddings[each[0]][1] * 1000)
            yyy.append(y1)
            x2 = float(embeddings[each[1]][0] * 1000)
            xxx.append(x2)
            y2 = float(embeddings[each[1]][1] * 1000)
            yyy.append(y2)
            ax.plot(xxx, yyy, '-', label='debug', linewidth=2, color='#2F54EB')

        ax.plot(xx, yy, 'o', label='debug', marker='o', markersize=15, color='#F64C4C')

        plt.show()

        pp = PdfPages(pdf_output)
        pp.savefig(fig)
        pp.close()




    @staticmethod
    def drawGNE():
        """
        visualization of the GNE model
        :return:
        """

        pdf_output = 'path of the result'

        f = open('path of GNE results which can be trained by project on github', encoding='utf-8')
        content = f.read()
        dict = json.loads(content)
        embeddings = dict['coordinates']

        tree, total_level, all_leaves = etl.prepare_tree('path to the tree file')

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        plt.style.use('bmh')

        xx = []
        yy = []
        for index in range(len(embeddings)):

            each = embeddings[index]
            x = each[0] * 1
            y = each[1] * 1
            x = float(x)
            y = float(y)
            xx.append(x)
            yy.append(y)

        edges_set = set()

        for node in tree:
            path = node.path
            for i in range(len(path) - 1):
                if((path[i],path[i+1]) not in edges_set):
                    edges_set.add((path[i],path[i+1]))


        for each in list(edges_set):

            xxx = []
            yyy = []
            x1 = float(embeddings[each[0]][0] * 1)
            xxx.append(x1)
            y1 = float(embeddings[each[0]][1] * 1)
            yyy.append(y1)
            x2 = float(embeddings[each[1]][0] * 1)
            xxx.append(x2)
            y2 = float(embeddings[each[1]][1] * 1)
            yyy.append(y2)
            ax.plot(xxx, yyy, '-', label='debug', linewidth=2, color='#2F54EB')


        ax.plot(xx, yy, 'o', label='debug', marker='o', markersize=20, color='#F64C4C')
        plt.style.use('bmh')
        plt.show()

        pp = PdfPages(pdf_output)
        pp.savefig(fig)
        pp.close()

    @staticmethod
    def reconstruction(embedding,target):
        """
        reconstruction for network
        :param embedding:
        :param target:
        :return:
        """

        singleDim = 32
        flag_file = f"../data/flag_{target}.txt"
        edge_file = f"../data/edges_{target}.txt"
        tree_file = f"../data/tree2_{target}"
        tree, total_level, all_leaves = etl.prepare_tree('../data/tree2_'+target)
        adjM = etl.prepare_graph(edge_file)
        objects = np.array(list(adjM.adj.keys()))
        nodesNum = len(objects)

        def calcLowerBoundDist(lowerBoundEmbedding,singleDim, needSum=True):
            """
            Calculate the distance based on the lower bound among embeddings.
            :param lowerBoundEmbedding:
            :return:
            """
            nodesNum = len(lowerBoundEmbedding)
            emb1 = torch.reshape(lowerBoundEmbedding.t(), (lowerBoundEmbedding.numel(), 1))
            emb1 = emb1.repeat(1, nodesNum)
            emb2 = torch.repeat_interleave(lowerBoundEmbedding.t(), repeats=nodesNum, dim=0)

            dimMixedDiff = torch.abs(emb1 - emb2)

            lowerDist = torch.unsqueeze(dimMixedDiff, 0).reshape(singleDim, nodesNum, nodesNum)
            if needSum:
                lowerDist = torch.sum(lowerDist, dim=0)

            return lowerDist


        def calcDist(tree, embedding):
            """
            distance metric
            :param tree:
            :param embedding:
            :return:
            """
            embeddingInTorch = torch.from_numpy(embedding)
            embeddingInTorchLower, embeddingInTorchHigher = torch.split(embeddingInTorch, singleDim, dim=1)
            layerBasedIndex = {}
            layerBasedTorch = {}
            for i in range(total_level):
                layerTmp = []
                for node in tree:
                    if node.level - 1 == i:
                        layerTmp.append(node.id)

                layerBasedTorch[i] = torch.index_select(
                    embeddingInTorchLower,
                    dim=0,
                    index=torch.tensor(layerTmp)
                )
                layerBasedIndex[i] = layerTmp

            layerBasedDist = {}
            for each in layerBasedTorch:
                layerBasedDist[each] = calcLowerBoundDist(layerBasedTorch[each], singleDim)

            leaves = layerBasedIndex[total_level-1]
            leavesNum = len(leaves)
            leavesParentDict = {}
            for eachLayer in range(total_level):
                leavesParentDict[eachLayer] = []
            for node in tree:
                if node.id < leavesNum:
                    for eachLayer in range(total_level):
                        leavesParentDict[eachLayer].append(node.path[eachLayer])
            pass
            leavesParentIndexList = {}
            for layerConuter in range(total_level):
                leavesParentIndexList[layerConuter] = [layerBasedIndex[layerConuter].index(x) for x in leavesParentDict[layerConuter]]

            finalDist = torch.zeros((leavesNum,leavesNum))
            for u in range(leavesNum):
                print('handling u:%d'%u)
                for v in range(u,leavesNum):
                    if u == v:
                        continue
                    for index in leavesParentIndexList:
                        if index == 0 :
                            continue
                        indexu = leavesParentIndexList[index][u]
                        indexv = leavesParentIndexList[index][v]
                        added = finalDist[u][v] + layerBasedDist[index][indexu][indexv]
                        finalDist[u][v] = finalDist[v][u] = added
            return finalDist

        ranksum = nranks = ap_scores = iters = 0
        labels = np.empty(nodesNum)
        distMatrix = calcDist(tree, embedding)
        for object in objects:
            labels.fill(0)
            neighbors = np.array(list(adjM.adj[object]))

            if (len(neighbors) == 0):
                continue

            objDist = distMatrix[object]
            objDist[object] = 1e5
            sorted_dists, sorted_idx = objDist.sort()
            ranks, = np.where(np.in1d(sorted_idx.detach().cpu().numpy(), neighbors))
            # The above gives us the position of the neighbors in sorted order.  We
            # want to count the number of non-neighbors that occur before each neighbor
            ranks += 1
            N = ranks.shape[0]

            # To account for other positive nearer neighbors, we subtract (N*(N+1)/2)
            # As an example, assume the ranks of the neighbors are:
            # 0, 1, 4, 5, 6, 8
            # For each neighbor, we'd like to return the number of non-neighbors
            # that ranked higher than it.  In this case, we'd return 0+0+2+2+2+3=14
            # Another way of thinking about it is to return
            # 0 + 1 + 4 + 5 + 6 + 8 - (0 + 1 + 2 + 3 + 4 + 5)
            # (0 + 1 + 2 + ... + N) == (N * (N + 1) / 2)
            # Note that we include `N` to account for the source embedding itself
            # always being the nearest neighbor
            ranksum += ranks.sum() - (N * (N - 1) / 2)
            nranks += ranks.shape[0]
            labels[neighbors] = 1
            ap_scores += average_precision_score(labels, -objDist.detach().cpu().numpy())
            iters += 1



