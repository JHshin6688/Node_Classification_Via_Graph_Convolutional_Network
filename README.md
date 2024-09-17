# Node_Classification_Via_Graph_Convolutional_Network

In this project, I built a GCN model that classifies nodes in the Cora dataset, using GraphSAGE and GAT layers.

For this project, I utilized the Cora dataset. Cora dataset consists of a citation network of scientific publications. Each publication (node) is connected by edges (citations) to other publications. The nodes are also labeled with classes that represent the topic of the paper.

I Implemented GraphSAGE and GAT layers from scratch, using Pytorch.

# GraphSAGE (Graph Sample and Aggregated)
GraphSAGE is a framework for generating node embeddings in large graphs. Unlike traditional approaches that require the entire graph to be processed simultaneously, GraphSAGE generates embeddings in a more scalable way by sampling and aggregating features from a node’s local neighborhood.

Instead of using all neighbors of a node, GraphSAGE samples a fixed number of neighbors. This helps in handling large graphs by limiting the computational cost.
GraphSAGE defines different aggregation functions to combine the features of the sampled neighbors. Common aggregation functions include:
- Mean Aggregator: Takes the mean of the neighbors' features.
- LSTM Aggregator: Uses an LSTM-based sequence model to aggregate neighbor features.
- Pooling Aggregator: Applies a pooling operation (e.g., max pooling) to the neighbors' features.
GraphSAGE is designed to be inductive, meaning it can generate embeddings for nodes that were not seen during training, making it suitable for dynamic and evolving graphs.

My GraphSAGE model is implemented in 'GraphSAGE.ipynb'.

More details can be found in this paper "Inductive Representation Learning on Large Graphs" https://arxiv.org/abs/1706.02216.


# GAT (Graph Attention Network)
GAT introduces the attention mechanism to graph neural networks, allowing nodes to assign different levels of importance to their neighbors when aggregating their features. This attention mechanism makes the model more flexible and powerful in capturing the structure of the graph.

Each node learns to weigh its neighbors' features differently based on their importance. GAT employs multi-head self-attention layers, similar to the attention mechanism in Transformers. Multiple attention heads can capture different aspects of the neighborhood.

Weighted neighbor features are aggregated using the attention coefficients, allowing the model to focus more on relevant neighbors. The aggregated features, combined with the node’s own features, are passed through a neural network layer to update the node’s embedding.

There are two types of models in 'GAT.ipynb'.
- 'GAT' is a static graph attention network.
- 'GATv2' is a dynamic graph attention network.

More details can be found in this paper "Graph Attention Networks" https://arxiv.org/abs/1710.10903.
