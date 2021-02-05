#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

import numpy as np
import math

import torch
import torch.nn as nn
from operator import itemgetter



class HierarchyModel(nn.Module):
    def __init__(self,pnode,omega,args,childrenList,res,device,parentDict, tree):
        super(HierarchyModel, self).__init__()
        self.parent = pnode
        self.args = args
        self.tree = tree
        self.childrenList = childrenList
        self.childrenNodesNum = len(childrenList)
        self.circleRange = args.single_circle_range
        self.hiddenDim = args.hidden_dim_t
        self.singleDim = args.single_dim_t
        self.device = device
        self.simOmegaMatrix = omega
        self.parentDict = parentDict
        self.res = res.to(device)
        # Number of leaves that derived from each child node.
        self.eachNodeLeavesNumCounter = []
        for child in childrenList:
            self.eachNodeLeavesNumCounter.append(len(tree[child].leaves))

        # A list of ratios.
        self.eachNodeLeavesNumRatio = torch.div(torch.Tensor(self.eachNodeLeavesNumCounter), sum(self.eachNodeLeavesNumCounter)).to(device)
        # Embedding initialization.
        self.parentEmbedding = res[pnode]
        correspondingParentsEmbL,correspondingParentsEmbH = torch.chunk(self.parentEmbedding, 2, dim=0)

        correspondingParentsEmbL = correspondingParentsEmbL.to(device)
        correspondingParentsEmbH = correspondingParentsEmbH.to(device)
        self.correspondingParentsEmbL_ = torch.add(correspondingParentsEmbL, self.circleRange).to(device)
        self.correspondingParentsEmbH_ = torch.add(correspondingParentsEmbH, self.circleRange).to(device)

        self.parentRange = HierarchyModel.clip_by_min(correspondingParentsEmbH - correspondingParentsEmbL, m=1e-5)

        initRangeForChildren = torch.mul(self.eachNodeLeavesNumRatio.unsqueeze(1), self.parentRange).to(device)

        # Initialize the embedding of the next layer.
        self.parent_embedding = res[pnode]
        parent_embedding_l, parent_embedding_h = torch.chunk(self.parent_embedding, 2, dim=0)

        for dim in range(self.singleDim):
            layerLowerEmbeddingE = torch.zeros(self.childrenNodesNum, 1).to(device)
            positive = max(parent_embedding_l[dim] , parent_embedding_h[dim])
            negative = min(parent_embedding_l[dim] , parent_embedding_h[dim])
            nn.init.uniform_(
                tensor=layerLowerEmbeddingE,
                a=negative+self.circleRange,
                b=positive+self.circleRange
            )

            if dim == 0:
                layerLowerEmbedding = layerLowerEmbeddingE
            else:
                layerLowerEmbedding = torch.cat((layerLowerEmbedding, layerLowerEmbeddingE), 1)

        layerHigherEmbedding = layerLowerEmbedding + initRangeForChildren

        self.childrenLowerEmbedding = nn.Parameter(layerLowerEmbedding, requires_grad=True)
        self.childrenHigherEmbedding = nn.Parameter(layerHigherEmbedding, requires_grad=True)



    def calcLayerBasedDist(self,layerBasedRes,res):
        """
        Add nodes' distance at this layer based on the lower bound.
        :param layerBasedRes:
        :param res:
        :return:
        """
        layerContains = self.parentsList
        layerContainsInRes = torch.index_select(
            res,
            dim=0,
            index=torch.tensor(layerContains).to(self.device)
        )
        layerContainsLower, layerContainsHigher = torch.split(layerContainsInRes, self.singleDim, dim=1)

        curLayerDist = self.calcLowerBoundDist(layerContainsLower,layerContainsHigher)

        # Calculate the number of children of each node in this layer.
        for parent in self.parentsList:
            self.childrenNumOfEachParent.append(len(self.tree[parent].direct_children))

        # Accumulate the upper layers' distance.
        if self.curLayer > 0:
            accumulatedLayerDist = curLayerDist + layerBasedRes[self.curLayer - 1]
            accumulatedLayerExpand = torch.repeat_interleave(accumulatedLayerDist, repeats=torch.tensor(self.childrenNumOfEachParent).to(self.device),dim=0)
            accumulatedLayerExpand = torch.repeat_interleave(accumulatedLayerExpand, repeats=torch.tensor(self.childrenNumOfEachParent).to(self.device),dim=1)
            layerBasedRes[self.curLayer] = accumulatedLayerExpand
        else:
            curLayerDistExpand = torch.repeat_interleave(curLayerDist, repeats=torch.tensor(self.childrenNumOfEachParent).to(self.device), dim=0)
            curLayerDistExpand = torch.repeat_interleave(curLayerDistExpand, repeats=torch.tensor(self.childrenNumOfEachParent).to(self.device), dim=1)
            layerBasedRes[self.curLayer] = curLayerDistExpand

        return layerBasedRes


    def calcLowerBoundDist(self, lowerBoundEmbedding, needSum = True):
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

        lowerDist = torch.unsqueeze(dimMixedDiff, 0).reshape(self.singleDim, nodesNum, nodesNum)
        if needSum:
            # Add all the dimensions.
            lowerDist = torch.sum(lowerDist, dim=0)

        return lowerDist



    def forward(self,idIndexes,omegaEmb,epoch):
        # Fix one part and train the other or conversely.
        # if epoch % 2 == 0:
        #     self.childrenLowerEmbedding.requires_grad_(True)
        #     self.childrenHigherEmbedding.requires_grad_(False)
        # else:
        #     self.childrenLowerEmbedding.requires_grad_(False)
        #     self.childrenHigherEmbedding.requires_grad_(True)
        lossDistance  =  lossShapeLike = lossExceed = lossOverlap = lossPositive = 0

        if epoch % 4 < 2:
            self.childrenLowerEmbedding.requires_grad_(True)
            self.childrenHigherEmbedding.requires_grad_(False)
        else:
            self.childrenLowerEmbedding.requires_grad_(False)
            self.childrenHigherEmbedding.requires_grad_(True)

        self.childrenEmbedding = torch.cat((self.childrenLowerEmbedding, self.childrenHigherEmbedding), 1)

        ids = [self.childrenList[i] for i in idIndexes]
        # number of nodes
        nodesNum = len(ids)

        omegaEmb4ids = omegaEmb
        finalEmb4ids = torch.index_select(
            self.childrenEmbedding,
            dim=0,
            index=idIndexes
        )

        childrenEmbeddingLower, childrenEmbeddingHigher = torch.split(finalEmb4ids, self.singleDim, dim=1)
        if self.args.loss_exceed > 0:
            # Calculate the penalty for the exceed part---start.
            exceedPart1_1 = self.correspondingParentsEmbL_ - childrenEmbeddingLower
            exceedPart1_2 = self.correspondingParentsEmbL_ - childrenEmbeddingHigher
            exceedPart2_1 = childrenEmbeddingHigher - self.correspondingParentsEmbH_
            exceedPart2_2 = childrenEmbeddingLower - self.correspondingParentsEmbH_
            lossExceed = torch.relu(exceedPart1_1).sum() + torch.relu(exceedPart2_1).sum() + torch.relu(exceedPart1_2).sum() + torch.relu(exceedPart2_2).sum()
            # Calculate the penalty for the exceed part---end.

        childrenEmbDiff = childrenEmbeddingHigher - childrenEmbeddingLower
        if self.args.loss_overlap > 0:
            # Calculate the penalty for the overlap part---start.
            childrenEmbeddingLowerTran1 = torch.reshape(childrenEmbeddingLower.t(),(childrenEmbeddingLower.numel(),1))
            childrenEmbeddingLowerTran1 = childrenEmbeddingLowerTran1.repeat(1, nodesNum)
            childrenEmbeddingHigherTran1 = torch.reshape(childrenEmbeddingHigher.t(),(childrenEmbeddingHigher.numel(),1))
            childrenEmbeddingHigherTran1 = childrenEmbeddingHigherTran1.repeat(1, nodesNum)
            childrenEmbeddingLowerTran2 = torch.repeat_interleave(childrenEmbeddingLower.t(), repeats=nodesNum, dim=0)
            childrenEmbeddingHigherTran2 = torch.repeat_interleave(childrenEmbeddingHigher.t(), repeats=nodesNum, dim=0)

            maxLower = torch.where(childrenEmbeddingLowerTran1 > childrenEmbeddingLowerTran2, childrenEmbeddingLowerTran1,
                                childrenEmbeddingLowerTran2)
            minHigher = torch.where(childrenEmbeddingHigherTran1 < childrenEmbeddingHigherTran2, childrenEmbeddingHigherTran1,
                                childrenEmbeddingHigherTran2)

            overlapPre = minHigher - maxLower
            overlapFilter = torch.ones((nodesNum, nodesNum)) - torch.eye((nodesNum))
            overlapFilter = overlapFilter.repeat(self.singleDim, 1).to(self.device)
            overlap_ = torch.mul(overlapPre, overlapFilter)

            overlap = torch.relu(overlap_).to(self.device)
            lossOverlap = overlap.sum()
            # Calculate the penalty for the overlap part---end.

        if self.args.loss_shape > 0:
            # Calculate the penalty for the shape-like part ---start.
            numeratorShapeLike = HierarchyModel.clip_by_min(torch.div(childrenEmbDiff, self.parentRange))
            denominatorShapeLike = torch.index_select(
                self.eachNodeLeavesNumRatio,
                dim=0,
                index=idIndexes
            )
            denominatorShapeLike = denominatorShapeLike.unsqueeze(1)
            shapeLikeDiv = torch.div(numeratorShapeLike, denominatorShapeLike)
            shapeLikeDiv = HierarchyModel.clip_by_max(shapeLikeDiv, ma=1.99, mi=0.01)
            lossShapeLike = torch.abs(torch.tan(torch.mul(torch.add(shapeLikeDiv, -1),math.pi / 2))).sum()
            # Calculate the penalty for the shape-like part ---end.

        if self.args.loss_distance > 0 :
            # Calculate the penalty for the distance part ---start.
            distBound = (childrenEmbeddingLower + childrenEmbeddingHigher) * 0.5
            realDistanceV2 = self.calcLowerBoundDist(distBound, needSum=False)
            realDistanceV2Square = realDistanceV2**2
            realDistanceV2SquareSum = torch.sum(realDistanceV2Square,dim=0)
            realDistanceV2SquareSumNormed = torch.norm(realDistanceV2SquareSum)
            distNormed = torch.div(realDistanceV2SquareSum, HierarchyModel.clip_by_min(realDistanceV2SquareSumNormed))
            omegaInnerProduct = torch.norm(omegaEmb4ids, dim=1, keepdim=True)
            omegaInnerProduct = torch.mul(omegaInnerProduct, omegaInnerProduct)
            omegaDist = -2 * torch.mm(omegaEmb4ids, omegaEmb4ids.t()) + omegaInnerProduct + omegaInnerProduct.t()
            omegaDist = HierarchyModel.clip_by_min(omegaDist).to(self.device)
            omegaDistNormed = omegaDist / HierarchyModel.clip_by_min(torch.norm(omegaDist))
            lossDistance = torch.norm(distNormed - omegaDistNormed)
            # Calculate the penalty for the distance part ---end.

        if self.args.loss_positive > 0:
            # Calculate the penalty for the gap part ---start.
            lossGap = torch.relu(self.parentRange * 1 - torch.sum(childrenEmbDiff, dim=0))
            # Calculate the penalty for the gap part ---end.
            lossPositive = lossGap.sum() + HierarchyModel.clip_by_min(torch.exp(-1 * (childrenEmbDiff))).sum()


        mark = epoch % 2

        # The following part can be modified flexibly
        if mark == 0:
            loss = self.args.loss_distance * lossDistance + self.args.loss_overlap * lossOverlap + self.args.loss_exceed * lossExceed + self.args.loss_shape * lossShapeLike
        else:
            loss = self.args.loss_positive * lossPositive



        return loss



    @staticmethod
    def trainStep(model, optimizer,treeIterator, step):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        optimizer.zero_grad()
        data = next(treeIterator)
        loss = model(data,step)
        loss.backward()
        optimizer.step()
        return loss.item(), model.childrenEmbedding

    @staticmethod
    def clip_by_min(x, m=1e-10):
        return torch.clamp(x, m, float('inf'))

    @staticmethod
    def clip_by_max(x, mi=-2, ma=1e5):
        return torch.clamp(x, mi, ma)