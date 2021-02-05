#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import math
import os
import time
import torch




from codes.network import NetworkModel
from codes.hierarchy import HierarchyModel
from codes.utils.datahandler import networkDataset,treeDataset
from codes.utils.datahandler import BidirectionalOneShotIterator
import codes.utils.data_etl as etl
from torch.utils.data import DataLoader
from codes.utils.filewriter import write_to_file
import numpy as np

def parse_args(args=None):
    """
    parse the parameters
    """
    parser = argparse.ArgumentParser(
        description='Train the HASNE Model'
    )

    parser.add_argument('--usecuda', action='store_true', help='use GPU')
    parser.add_argument('--calc_sim_from_source', action='store_true', help='use GPU')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--target', type=str, default="hamilton")
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--max_epoch', type=int, default=50000)
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    parser.add_argument('--scr_times', type=int, default=10)
    parser.add_argument('-dn', '--hidden_dim_n', default=16, type=int)
    parser.add_argument('-dt', '--hidden_dim_t', default=2, type=int)
    parser.add_argument('--cpu_num', type=int, default=1)
    parser.add_argument('-cm','--circle_margin', type=float)
    parser.add_argument('-b', '--batch_size', default=500, type=int)
    parser.add_argument('--res_path', type=str, default='../res')
    parser.add_argument('--loss_distance', default=0.0001, type=float)
    parser.add_argument('--loss_shape', default=0.0001, type=float)
    parser.add_argument('--loss_overlap', default=0.0001, type=float)
    parser.add_argument('--loss_exceed', default=0.0001, type=float)
    parser.add_argument('--loss_positive', default=0.0001, type=float)

    return parser.parse_args(args)



def nodeWiseTraining(curNode, res, args, tree, leavesMatrix, device, layerCounter, parentDict, layerBasedDict, mediumRes):
    """
    Node by node training function.
    :param curNode:  current node that to be trained
    :param res:         the final results
    :param args:        the arguments in script
    :param tree:        the tree structure
    :param leavesMatrix:    the similarity matrix among leaves
    :param device:      the training device
    :param layerCounter:    a dict that store the list of nodes of each layer
    :return:
    """
    childrenList, simMatrix = etl.getNodesSimBasedOnLeavesSim(curNode, tree, leavesMatrix, 100)
    layer = tree[curNode].level
    # Return the leaf level and needn't process.
    if len(childrenList) < 2:
        res[childrenList[0]] = res[curNode] + args.single_circle_range
        mediumRes[childrenList[0]] = mediumRes[curNode]

    else:
        print("Start training the network: Node: "+str(curNode)+"......")

        # Calc the ground-truth of this layer, the simMatrix is in the order of childrenList.
        simMatrixNorm = etl.normalizeMatrix(simMatrix)

        # Initialize the network model.
        networkModel = NetworkModel(
            children=childrenList,
            args=args
        )
        # Load the network training dataset.
        networkDataLoader = DataLoader(
            networkDataset(simMatrixNorm, args, childrenList),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=lambda x: networkDataset.collate_fn(x, args.batch_size),
            drop_last=False
        )
        # The network iterator.
        networkTrainingIterator = BidirectionalOneShotIterator(networkDataLoader)

        networkLearningRate = args.learning_rate
        networkOptimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, networkModel.parameters()),
            lr=networkLearningRate
        )
        # Start training the network.
        preLoss = float('inf')
        for step in range(0, args.max_steps):
            loss, embeddingOmega = NetworkModel.train_step(networkModel, networkOptimizer, networkTrainingIterator)
            if step % 100 == 0:
                lossNumeric = loss.item()
                print("Network layer:%d, iterator is %d, loss is:%f" % (curNode, step, lossNumeric))
                if abs(lossNumeric - preLoss) < 100:
                    break
                else:
                    preLoss = lossNumeric

        # Start training the tree.
        print("Start training the tree: Node: " + str(curNode) + "......")

        omega = embeddingOmega.data.numpy()

        treeModel = HierarchyModel(
            pnode = curNode,
            omega=omega,
            res=res,
            args=args,
            childrenList=childrenList,
            device=device,
            parentDict = parentDict,
            tree = tree
        )

        treeModel.to(device)
        treeModel.train()

        # Load the tree training dataset.
        treeDataLoader = DataLoader(
            treeDataset(omega, args),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=lambda x: treeDataset.collate_fn(x, args.batch_size),
            drop_last=False
        )

        treeLearningRate = args.learning_rate
        treeOptimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, treeModel.parameters()),
            lr=treeLearningRate
        )

        # Start training the tree in epochs.
        treePreLoss = float('inf')

        for epoch in range(0,  args.max_epoch * (10 - layer)):
            for i, data in enumerate(treeDataLoader):
                idx = data[0].to(device)
                omega = data[1].to(device)
                treeOptimizer.zero_grad()
                treeLoss = treeModel(idx, omega, epoch)
                treeLoss.backward()
                treeOptimizer.step()

            t = epoch % 100
            if t ==0:
                loss = treeLoss.item()
                print("Tree node:%d, epoch is %d, loss is:%f" % (curNode, epoch, loss))
                # stop training threshold.
                if abs(loss - treePreLoss) < 0.1 :
                    embTmpRes = treeModel.childrenEmbedding.data.cpu()

                    for indexer in range(len(childrenList)):
                        child = childrenList[indexer]
                        res[child] = embTmpRes[indexer]
                        mediumRes[child] = omega[indexer]
                    break
                else:
                    treePreLoss = loss
            # train to the last epoch, stop it.
            if epoch == args.max_epoch - 1:
                embTmpRes = treeModel.childrenEmbedding.data.cpu()
                for indexer in range(len(childrenList)):
                    child = childrenList[indexer]
                    res[child] = embTmpRes[indexer]
                    mediumRes[child] = omega[indexer]
    # DFS train its children.
    for child in childrenList:
        if tree[child].direct_children:
            nodeWiseTraining(child, res, args, tree, leavesMatrix, device, layerCounter, parentDict, layerBasedDict, mediumRes)



def main(args):
    """
    main entrance of the training
    :param args:
    :return:
    """
    # Parameters verifying.
    if (not args.do_train):
        raise ValueError('error.')
    if (args.hidden_dim_t % 2 != 0):
        raise ValueError('hidden_error')
    args.single_dim_t = args.hidden_dim_t // 2

    args.network_path = '../data/edges_'+args.target+'.txt'
    args.data_path = '../data/tree2_'+args.target
    args.save_sim_path = '../data/similarity_mat_'+args.target+'.txt'

    # define the size of one whole circumference
    args.single_circle_range = 2 * math.pi * args.scr_times
    # Select a device, cpu or gpu.
    if args.usecuda:
        devicePU = "cuda:3" if torch.cuda.is_available() else "cpu"
    else:
        devicePU = "cpu"
    device = torch.device(devicePU)

    # Load the tree and some properties of the tree.
    tree, total_level, all_leaves = etl.prepare_tree(args.data_path)

    # load the graph
    graph = etl.prepare_graph(args.network_path)

    # Define the root node
    root = len(tree) - 1

    # Calc the graph similarity, i.e. the matrix \capA in paper.
    # Since the matrix depends on the dataset, we needn't calculate twice.
    if args.calc_sim_from_source:
        leavesSimilarity = etl.get_leaves_similarity(graph)
        leavesSimilarity = np.array(leavesSimilarity)
        # save similarity matrix
        np.savetxt(args.save_sim_path, leavesSimilarity)
    else:
        # read the matrix from file.
        leavesSimilarity = np.loadtxt(args.save_sim_path)

    # Initialize the result and fix the root node's embedding.
    root_embedding_lower = torch.zeros(1, args.single_dim_t)
    root_embedding_upper = args.single_circle_range * torch.ones(1, args.single_dim_t)
    root_embedding = torch.cat((root_embedding_lower, root_embedding_upper), 1)[0]
    # The intermediate embedding.
    mediumRes = torch.zeros(len(tree), args.hidden_dim_n)
    res = torch.zeros(len(tree), args.hidden_dim_t)
    res[root] = root_embedding

    # Initialize the layer dict containing lists of nodes of each layer.
    layerCounter = [[] for i in range(total_level)]
    # A dictionary for parent-children relations.
    parentDict = {}
    for node in tree:
        if node.id != root:
            parentDict[node.id] = node.path[-2]
        layerCounter[node.level - 1].append(node.id)

    # Initialize the layerBasedDistance dict.
    layerBasedDict = {}

    # Train HASNE layer by layer, start from the 0(which the root locate in) layer.
    nodeWiseTraining(root, res, args, tree, leavesSimilarity, device, layerCounter, parentDict, layerBasedDict, mediumRes)

    # The path of the final output.
    res_output = os.path.join(args.res_path, "res_"+str(int(time.time())) + "_"+args.target + "_t"+str(args.hidden_dim_t)+"_n"+str(args.hidden_dim_n))

    # We record all the details in the finishing training.
    final_res = {
        'target':args.target,
        'learningRate':args.learning_rate,
        'batchSize':args.batch_size,
        'networkDims':args.hidden_dim_n,
        'treeDims':args.hidden_dim_t,
        'circleRange':args.single_circle_range,
        'epoch':args.max_epoch,
        'loss_distance':args.loss_distance,
        'loss_exceed':args.loss_exceed,
        'loss_overlap':args.loss_overlap,
        'loss_shape':args.loss_shape,
        'loss_positive':args.loss_positive,
        'embedding':res.numpy().tolist(),
        'embeddingOmega':mediumRes.numpy().tolist()
    }
    write_to_file(res_output, json.dumps(final_res))



if __name__ == '__main__':
    main(parse_args())
