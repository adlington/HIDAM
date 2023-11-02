import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNDGLModel(nn.Module):
    """
    Description
    -----------
    Base class for GNN models, which contains "fit" method for training and "predict" method for inference.

    Parameters
    ----------
    binary: bool, whether the model is a binary classification. (Default: True)
    target: str or None, the target node type in heterogeneous graphs. If None, create a homogeneous GNN model. (Default: None) 
    """
    def __init__(self, binary=True, target=None):
        super(GNNDGLModel,self).__init__()
        self.binary = binary
        self.target = target

    def fit(self, dataloader, inputs_all, label, optimizer, criterion, device=None, log=0, **kwargs):
        """
        Description
        -----------
        Train the GNN model with given dataloader.

        Parameters
        ----------
        dataloader: dgl.dataloading.NodeDataLoader, dataloader for batch-iterating over a set of training nodes, 
        generating the list of message flow graphs (MFGs) as computation dependency of the said minibatch.
        inputs_all: torch.Tensor or Dict[str, torch.Tensor], the features of all target type of nodes in the graph.
        label: torch.Tensor or Dict[str, torch.Tensor], the labels of all target type of nodes in the graph.
        optimizer: torch.optim.Optimizer, the optimizer for training.
        criterion: torch.nn.Module, the loss function for training.
        device: torch.device or None, if None, use cpu training. (Default: None)
        log: int, the number of batches to print the training log, if set to zero, no log prints. (Default: 0)
        **kwargs: other forward parameters of the model.
        """
        self.train()
        total_loss = 0
        datasize = len(dataloader.collator.dataset)
        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            if self.target is None:
                inputs = inputs_all[input_nodes]           
            else:
                if isinstance(input_nodes, dict):
                    input_nodes = input_nodes[self.target]
                if isinstance(output_nodes, dict):
                    output_nodes = output_nodes[self.target]
                inputs = {self.target: inputs_all[self.target][input_nodes]}
            y = label[output_nodes]
            if device is not None:
                if self.target is None:
                    inputs = inputs.to(device)
                else:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                y = y.to(device)
                blocks = [block.to(device) for block in blocks]
            logits = self.forward(blocks, inputs, **kwargs)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate loss and accuracy if log is required
            if log > 0:
                total_loss += loss.item() * len(output_nodes)
                acc = (logits.argmax(1) == y).float().mean()
                train_size = len(output_nodes)*(step+1) if len(output_nodes)*(step+1) < datasize else datasize

                if (step+1) % log == 0:
                    print("step:{},data:{}/{},loss:{:.4f},acc:{:.4f}".format(step+1,train_size,datasize,total_loss,acc))

    def predict(self, dataloader, inputs_all, label, criterion, device=None, **kwargs):
        """
        Description
        -----------
        Predict the GNN model with given dataloader.

        Parameters
        ----------
        dataloader: dgl.dataloading.NodeDataLoader, dataloader for batch-iterating over a set of nodes, 
        generating the list of message flow graphs (MFGs) as computation dependency of the said minibatch.
        inputs_all: torch.Tensor or Dict[str, torch.Tensor], the features of all target type of nodes in the graph.
        label: torch.Tensor or Dict[str, torch.Tensor], the labels of all target type of nodes in the graph.
        criterion: torch.nn.Module, the loss function.
        device: torch.device or None, if None, use cpu for inference. (Default: None)
        **kwargs: other forward parameters of the model.

        Returns
        -------
        total_loss: float, average loss.
        y_prob: List[float] or List[int], positive probability or predicted labels of target nodes.
        """
        self.eval()
        total_loss = 0
        y_pred_list = []
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in dataloader:
                if self.target is None:
                    inputs = inputs_all[input_nodes]
                else:
                    if isinstance(input_nodes, dict):
                        input_nodes = input_nodes[self.target]
                    if isinstance(output_nodes, dict):
                        output_nodes = output_nodes[self.target]
                    inputs = {self.target: inputs_all[self.target][input_nodes]}
                y = label[output_nodes]
                if device is not None:
                    if self.target is None:
                        inputs = inputs.to(device)
                    else:
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    y = y.to(device)
                    blocks = [block.to(device) for block in blocks]
                logits = self.forward(blocks, inputs, **kwargs)
                loss = criterion(logits, y)
                total_loss += loss.item() * len(output_nodes)
                if self.binary:
                    y_pred_list.append(F.softmax(logits, dim=1)[:, 1].cpu())
                else:
                    y_pred_list.append(logits.argmax(1).cpu())

            total_loss /= len(dataloader.collator.dataset)
            y_prob = torch.cat(y_pred_list, dim=0)

            return total_loss, y_prob

class classifier(nn.Module):
    """
    Description
    -----------
    A MLP classifier.

    Parameters
    ----------
    in_dim: int, input layer size.
    hid_dim: int, hidden layer size.
    out_dim: int, output layer size.
    num_layers: int, number of layers. (Default: 2)
    activation: func or None, activation function, if None, no activation. (Default: F.relu)
    normalize: bool, whether the model applies batch normalization layers. (Default: True)
    dropout: float, the dropout rate in hidden layers. (Default: 0.0)
    """
    def __init__(self,
                in_dim,
                hid_dim,
                out_dim,
                num_layers = 2,
                activation = F.relu,
                normalize = True,
                dropout = 0.0):
        super(classifier,self).__init__()
        assert num_layers >= 2, "the number of layers must be at least 2!"
        self.activation = activation
        self.norm = normalize
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.ModuleList()
        self.clf.append(nn.Linear(in_dim, hid_dim))
        for i in range(num_layers-2):
            self.clf.append(nn.Linear(hid_dim, hid_dim))
        self.clf.append(nn.Linear(hid_dim, out_dim))
        if normalize:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hid_dim) for i in range(num_layers-1)])
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        for l0 in self.clf:
            l0.reset_parameters()
        if self.norm:
            for l1 in self.bn:
                l1.reset_parameters()

    def forward(self, x):
        """
        Description
        -----------
        Forward propagation calculation for the model.

        Parameters
        ----------
        x: torch.Tensor, input features.

        Returns
        -------
        x: torch.Tensor, output results.
        """
        if self.norm:
            for l, b in zip(self.clf[:-1], self.bn):
                x = self.dropout(self.activation(b(l(x))))
        else:
            for l in self.clf[:-1]:
                x = self.dropout(self.activation(l(x)))
        
        x = self.clf[-1](x)
        return x