import os
import torch

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch.nn import functional as F
from tqdm import trange

os.environ["TORCH"] = torch.__version__
print(torch.__version__)

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.scatter(z[:,0], z[:,1], s=70, c=color, cmap="Set2")

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, hidden_channels: int, dataset: Planetoid):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def train_epoch(self, optimizer, criterion, data):
        self.train()
        optimizer.zero_grad()
        out = self(data.x)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward() # Compute gradients
        optimizer.step()
        return loss

    def test(self, data):
        self.eval()
        out = self(data.x)
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels: int, dataset: Planetoid):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def train_epoch(self, optimizer, criterion, data):
        self.train()
        optimizer.zero_grad()
        out = self(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward() # Compute gradients
        optimizer.step()
        return loss

    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc

if __name__ == "__main__":
    dataset = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures())

    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

    data = dataset[0]

    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    model = MultiLayerPerceptron(4, dataset)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print()
    print(model)
    print()

    pbar = trange(1, 201)
    for epoch in pbar:
        loss = model.train_epoch(optimizer, criterion, data)
        pbar.set_description(f"Epoch: {epoch:03d} Loss: {loss:.4f}")

    test_acc = model.test(data)
    print(f"Test accuracy: {test_acc}")

    gcn_model = GCN(16, dataset)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)

    print()
    print(gcn_model)
    print()

    gcn_model.eval()
    out = gcn_model(data.x, data.edge_index)
    plt.subplot(1, 2, 1)
    visualize(out, color=data.y)
    plt.title("Before Training")

    pbar = trange(1, 101)
    for epoch in pbar:
        loss = gcn_model.train_epoch(optimizer, criterion, data)
        pbar.set_description(f"Epoch: {epoch:03d} Loss: {loss:.4f}")

    test_acc = gcn_model.test(data)
    print(f"Test accuracy: {test_acc}")

    gcn_model.eval()
    out = gcn_model(data.x, data.edge_index)
    plt.subplot(1, 2, 2)
    visualize(out, color=data.y)
    plt.title(f"After Training (acc={test_acc})")

    plt.show()

