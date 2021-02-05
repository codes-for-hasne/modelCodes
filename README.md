# HASNE

Pytorch implementation of HASNE.

#### Usage:

```
python3 ./run.py {Parameters for the model. An example is shown in example.params.txt}
```

#### Explanation for dataset:

##### Datasets from Facebook-100:

- "edges_xxx.txt" saves the edges in a network;
- "Tree2_xxx" saves the hierarchy of the xxx network;
- "Flag_xxx.txt" saves the labels of the nodes;

##### Datasets from HRG:

​	We follow these projects to consturct the HRG_n datasets and the scripts for data processing are in the `data` directory. 

- https://github.com/yasirs/cmn
- aaronclauset.github.io

#### Codes of Poincare and GNE

- https://github.com/facebookresearch/poincare-embeddings

- https://radimrehurek.com/gensim/models/poincare.html

- https://github.com/lundu28/GNE





