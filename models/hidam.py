import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn
from .base import GNNDGLModel, classifier


class InstanceFusion(nn.Module):
    """
    Description
    -----------
    Instance-level fusion layer.

    Parameters
    ----------
    out_dim: int, output node embedding size.
    target: str, the target node type.
    ntype_dict: dict[int, str], node type dict where key and value represent the index and the name of the node type respectively.
    etype_dict: dict[int, str], edge type dict where key and value represent the index and the name of the edge type respectively.
    mp_info: [torch.Tensor, torch.Tensor], the node type index and the edge type index along the metapath.
    edge_dim: int, the virtual edge feature size, if zero, no edge feature. (Default: 0)
    activation: func, activation function. (Default: F.relu)
    feat_dropout: float, input feature dropout rate. (Default: 0.0)
    attn_dropout: float, attention weights dropout rate. (Default: 0.0)
    normalize: bool, whether to use l2 normalization before output. (Default: True)
    residual: bool, whether to use residual connection. (Default: True)
    num_heads: int, number of heads in multi-heads attention. (Default: 1)
    """
    def __init__(self,
                out_dim,
                target,
                ntype_dict,
                etype_dict,
                mp_info,
                edge_dim = 0,
                activation = F.relu,
                feat_dropout = 0.0,
                attn_dropout = 0.0,
                normalize = True,
                residual = True,
                num_heads = 1
                ):
        super(InstanceFusion, self).__init__()
        self.out_dim = out_dim
        self.target = target
        self.ntype_dict = ntype_dict
        self.etype_dict = etype_dict
        self.mp_info = mp_info
        self.edge_dim = edge_dim
        self.activation = activation
        self.feat_dropout = nn.Dropout(feat_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.norm = normalize
        self.residual = residual
        self.num_heads = num_heads
        self.attn_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim)))
        self.attn_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim)))
        if edge_dim:
            self.attn_edge = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_dim)))
            self.fc_edge = nn.Linear(edge_dim, out_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        gain = nn.init.calculate_gain(self.activation.__name__)
        nn.init.xavier_normal_(self.attn_src, gain=gain)
        nn.init.xavier_normal_(self.attn_dst, gain=gain)
        if self.edge_dim:
            nn.init.xavier_normal_(self.attn_edge, gain=gain)
            self.fc_edge.reset_parameters()

    def forward(self, g, h, tf_mods, nfeat_dict, efeat_dict):
        """
        Description
        -----------
        Forward computation.

        Parameters
        ----------
        g: dgl.DGLHeteroGraph, the graph data.
        h: Tensor, the feature tensor of the target node type.
        tf_mods: dict[str, nn.Linear], the feature transformation matrix of different node and edge types.
        nfeat_dict: dict[str, Tensor], the feature tensors of different node types.
        efeat_dict: dict[str, Tensor], the feature tensors of different edge types.

        Returns
        -------
        out: Tensor, the output tensor.
        """
        with g.local_scope():
            if isinstance(h, tuple):
                h_src, h_dst = h
                h_src = tf_mods[self.target](self.feat_dropout(h_src)).view(-1, self.num_heads, self.out_dim)  # (num_src_node, num_heads, out_dim)
                h_dst = tf_mods[self.target](self.feat_dropout(h_dst)).view(-1, self.num_heads, self.out_dim)  # (num_src_node, num_heads, out_dim)
            else:
                h = tf_mods[self.target](self.feat_dropout(h)).view(-1, self.num_heads, self.out_dim)  # (num_src_node, num_heads, out_dim)
                if g.is_block:
                    h_src = h
                    h_dst = h[:g.number_of_dst_nodes()]
                else:
                    h_src = h_dst = h
            e_src = (self.attn_src * self.activation(h_src)).sum(dim=-1).unsqueeze(-1)  # (num_src_node, num_heads, 1)
            e_dst = (self.attn_dst * self.activation(h_dst)).sum(dim=-1).unsqueeze(-1)  # (num_dst_node, num_heads, 1)
            g.srcdata.update({"ft": h_src, "es": e_src})
            g.dstdata.update({"ed": e_dst})
            g.apply_edges(fn.u_add_v("es", "ed", "e"))  # (num_edge, num_heads, 1)
            # transform and concatenate the features of the specific node and edge types along the metapaths
            if self.edge_dim:
                feat_virtual = []
                for i in range(len(self.mp_info[0])):
                    nid = self.mp_info[0][i].item()
                    if self.ntype_dict[nid] in nfeat_dict:
                        # h_tmp = self.feat_dropout(nfeat_dict[self.ntype_dict[nid]][g.edata["hn"][:, i]])
                        # h_tmp = tf_mods[self.ntype_dict[nid]](h_tmp).view(-1, self.num_heads, self.out_dim)  # (num_edge, num_heads, out_dim)
                        # first transformed to hidden size to reduce memory cost
                        h_all = self.feat_dropout(nfeat_dict[self.ntype_dict[nid]])
                        h_all = tf_mods[self.ntype_dict[nid]](h_all).view(-1, self.num_heads, self.out_dim)  # (num_edge, num_heads, out_dim)
                        h_tmp = h_all[g.edata["hn"][:, i]]
                        feat_virtual.append(h_tmp)
                for i in range(len(self.mp_info[1])):
                    eid = self.mp_info[1][i].item()
                    if self.etype_dict[eid] in efeat_dict:
                        # e_tmp = self.feat_dropout(efeat_dict[self.etype_dict[eid]][g.edata["he"][:, i]])
                        # e_tmp = tf_mods[self.etype_dict[eid]](e_tmp).view(-1, self.num_heads, self.out_dim)  # (num_edge, num_heads, out_dim)
                        # first transformed to hidden size to reduce memory cost
                        e_all = self.feat_dropout(efeat_dict[self.etype_dict[eid]])
                        e_all = tf_mods[self.etype_dict[eid]](e_all).view(-1, self.num_heads, self.out_dim)  # (num_edge, num_heads, out_dim)
                        e_tmp = e_all[g.edata["he"][:, i]]
                        feat_virtual.append(e_tmp)
                
                g.edata["h"] = torch.cat(feat_virtual, dim=-1)  # (num_edge, num_heads, edge_dim)
                e_edge = (self.attn_edge * self.activation(g.edata["h"])).sum(dim=-1).unsqueeze(-1)  # (num_edge, num_heads, 1)
                g.edata["e"] = g.edata.pop("e") + e_edge
            # compute attention weights
            g.edata["a"] = self.attn_dropout(dglnn.functional.edge_softmax(g, g.edata.pop("e")))  # (num_edge, num_heads,1)
            g.update_all(message_func=fn.u_mul_e("ft", "a", "msg"), reduce_func=fn.sum("msg", "ft"))
            rst = g.dstdata["ft"]  # (num_dst_node, num_heads, out_dim)
            if self.edge_dim:
                g.update_all(message_func=lambda x: {"msg2": x.data["h"]*x.data["a"]},
                            reduce_func=fn.sum("msg2", "ft2"))  # (num_dst_node, num_heads, edge_dim)
                rst = rst + self.fc_edge(g.dstdata["ft2"])  # (num_dst_node, num_heads, out_dim)
            if self.residual:
                rst = rst + h_dst  # (num_dst_node, num_heads, out_dim)
            if self.activation:
                rst = self.activation(rst)
            out = rst.view(rst.shape[0], -1)  # (num_dst_node, num_heads*out_dim)
            if self.norm:
                out = F.normalize(out, p=2, dim=1)  # (num_dst_node, num_heads*out_dim)
            return out

class SemanticFusion(nn.Module):
    """
    Description
    -----------
    Metapath attention layer.

    Parameters
    ----------
    in_dim: int, the input feature.
    hid_dim: int, the metapath attention vector size. (Default: 128)
    """
    def __init__(self, in_dim, hid_dim=128):
        super(SemanticFusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1, bias=False)
        )

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        for l in self.project:
            if isinstance(l, nn.Linear):
                l.reset_parameters()

    def forward(self, h):
        """
        Description
        -----------
        Forward computation.

        Parameters
        ----------
        h: Tensor, the input feature tensor.

        Returns
        -------
        out: Tensor, the output feature tensor.
        """
        w = self.project(h)                    # (batch_size, num_metapath, 1)
        w = F.softmax(w, dim=1)                 # (batch_size, num_metapath, 1)
        out = (w * h).sum(1)                       # (batch_size, in_dim)
        return out

class HIDAM(GNNDGLModel):
    """
    Description
    -----------
    The proposed HIDAM model.

    Parameters
    ----------
    in_dim_dict: dict[str, int], input feature size of different node and edge types.
    hid_dim: int, the embedding size.
    out_dim: int, the output size.
    metapath_info: dict[str, dict[str, any]], the index and feature size along each metapath.
    target: str, the target node type.
    ntype_dict: dict[int, str], node type dict where key and value represent the index and the name of the node type respectively.
    etype_dict: dict[int, str], edge type dict where key and value represent the index and the name of the edge type respectively.
    activation: func, activation function. (Default: F.relu)
    dropout: float, dropout rate in the instance-level fusion layer and the MLP classifier. (Default: 0.0)
    l2_norm: bool, whether to use L2 normalization in the instance-level fusion layer. (Default: True)
    residual: bool, whether to use residual connection in the instance-level fusion layer. (Default: True)
    attn_dim: int, the dimension of the semantic attention vector. (Default: 128)
    batch_norm: bool, whether to use batch normalization layer in the MLP classifier. (Default: True)
    """
    def __init__(self,
                in_dim_dict, 
                hid_dim, 
                out_dim, 
                metapath_info, 
                target,
                ntype_dict,
                etype_dict,
                activation = F.relu, 
                dropout = 0.0,
                l2_norm = True,
                residual = True,
                attn_dim = 128,
                batch_norm = True
                ):
        super(HIDAM, self).__init__(target=target, binary=(out_dim==2))
        self.tf_mods = nn.ModuleDict({k: nn.Linear(v, hid_dim) for k, v in in_dim_dict.items()})
        self.conv = dglnn.HeteroGraphConv(
        {metapath: InstanceFusion(out_dim=hid_dim, target=target, ntype_dict=ntype_dict, etype_dict=etype_dict, mp_info=v["type"], edge_dim=v["dim"], 
        activation=activation, feat_dropout=dropout, attn_dropout=dropout, normalize=l2_norm, residual=residual) for metapath, v in metapath_info.items()},
        aggregate = "stack"
        )
        self.mp_attn = SemanticFusion(hid_dim, attn_dim)
        self.clf = classifier(hid_dim, hid_dim, out_dim, 2, activation, batch_norm, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        for obj, tf in self.tf_mods.items():
            tf.reset_parameters()
        for metapath, layer in self.conv.mods.items():
            layer.reset_parameters()
        self.mp_attn.reset_parameters()
        self.clf.reset_parameters()

    def forward(self, g, h, nfeat_dict, efeat_dict):
        """
        Description
        -----------
        Forward computation.

        Parameters
        ----------
        g: dgl.DGLHeteroGraph or List[MFG], the graph data.
        h: dict[str, Tensor], the feature tensor of the target node type.
        nfeat_dict: dict[str, Tensor], the feature tensors of different node types.
        efeat_dict: dict[str, Tensor], the feature tensors of different edge types.

        Returns
        -------
        out: Tensor, the output tensor.
        """
        mod_kwargs = {metapath: {"tf_mods": self.tf_mods, "nfeat_dict": nfeat_dict, "efeat_dict": efeat_dict} for metapath in self.conv.mods.keys()}
        if isinstance(g, list):
            for block in g:
                h = self.conv(block, h, mod_kwargs=mod_kwargs)  # (batch_size, num_metapath, hid_dim)
        else:
            h = self.conv(g, h, mod_kwargs=mod_kwargs)

        out = h[self.target]
        out = self.mp_attn(out)  # (batch_size, num_heads*hid_dim)
        out = self.clf(out)
        
        return out
