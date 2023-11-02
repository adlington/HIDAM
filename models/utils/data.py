import torch
import dgl
import os
import os.path as osp
import json
import pandas as pd
from dgl.data.utils import save_graphs,load_graphs
from sklearn.model_selection import train_test_split


def str_to_tensor(s):
    """
    Description
    -----------
    Convert features from str to tensor.

    Parameters
    ----------
    s: str, the input feature of str type.

    Returns
    -------
    torch.Tensor, feature tensor.
    """
    return torch.FloatTensor([float(i) for i in s.split(",")])

def neighbor_sampling(df, nsamples=20):
    """
    Description
    -----------
    Sampling path instances for each dst node.

    Parameters
    ----------
    df: DataFrame, data of all path instances.
    nsamples: int, sampling num.

    Returns
    -------
    result: DataFrame, data of sampling path instances.
    """
    df["rand"] = torch.randn((df.shape[0],1)).numpy()
    df["rank"] = df.groupby(["dst_idx"])["rand"].rank()
    result = df[df["rank"] <= nsamples].drop(columns=["rand", "rank"])
    return result

class DBLPMPGraph:
    """
    Description
    -----------
    Read graph data and establish the metapath-based neighbor graph, where the intermediate node and edge ids are kept in the attributes("hn" and "he") of the new edges.

    Parameters
    ----------
    root: str, the file directory.
    metapath_list: List[List[str]], lists of metapaths.
    sampling: int, sampling number of path instances. If set to 0, no sampling. (Default: 0)
    """
    def __init__(self, root, metapath_list, sampling=0):
        self.root = root
        self.metapath_list = metapath_list
        self.sampling = sampling
        # self.raw_dir = osp.join(root, "raw")
        self.raw_dir = root
        self.processed_dir = osp.join(root, "processed")
        self.info()
        self.process()

    def info(self):
        """
        Description
        -----------
        Get the graph info.
        """
        info = json.loads(open(osp.join(self.raw_dir, "info.dat")).read())
        self.node_info = info["node.dat"]["node type"]
        self.edge_info = info["link.dat"]["link type"]
        self.target = list(info["label.dat"]["node type"].keys())[0]
        meta = json.loads(open(osp.join(self.raw_dir, "meta.dat")).read())
        self.num_nodes_dict = {}
        for k, v in self.node_info.items():
            self.num_nodes_dict[v] = int(meta["Node Type_" + k])

    def process(self):
        """
        Description
        -----------
        Read graph data and get the metapath-based neighbor graph.
        """
        if not osp.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        processed_file_graph = osp.join(self.processed_dir, "DBLP_mp_{}.bin".format(self.sampling))
        processed_file_info = osp.join(self.processed_dir, "DBLP_mp_{}.pth".format(self.sampling))
        # load data from saved file
        if osp.exists(processed_file_graph):
            self.load(processed_file_graph, processed_file_info)
        else:
            # get node features
            nodedf = pd.read_csv(osp.join(self.raw_dir, "node.dat"), names=["idx", "name", "ntype", "feat"], sep="\t")
            # re-index different types of nodes
            nodedf["new_idx"] = nodedf.groupby(["ntype"])["idx"].rank().astype("int") - 1
            # save node features of different types
            self.nfeat = {}
            ntypelist = nodedf[nodedf.feat.notnull()].ntype.unique().tolist()
            for ntype in ntypelist:
                node_tmp = nodedf[nodedf.ntype == ntype].feat.tolist()
                self.nfeat[self.node_info[str(ntype)]] = torch.cat([str_to_tensor(v).unsqueeze(0) for v in node_tmp], dim=0).to(torch.float32)

            # read edges
            edgedf = pd.read_csv(osp.join(self.raw_dir, "link.dat"), names=["src", "dst", "etype", "weight"], sep="\t")
            # save edge features of different types
            self.efeat = {}
            # reorganize the edge data in a dict
            data_dict = {}
            for etype, edict in self.edge_info.items():
                edge_tmp = edgedf[edgedf.etype == int(etype)][["src", "dst"]]
                src_tmp = nodedf[nodedf.ntype == int(edict["start"])][["idx", "new_idx"]].rename(columns={"idx":"src", "new_idx":"src_idx"})
                edge_tmp = pd.merge(edge_tmp, src_tmp, on=["src"], how="left")
                dst_tmp = nodedf[nodedf.ntype == int(edict["end"])][["idx", "new_idx"]].rename(columns={"idx":"dst", "new_idx":"dst_idx"})
                edge_tmp = pd.merge(edge_tmp, dst_tmp, on=["dst"], how="left")
                edge_tmp = edge_tmp.reset_index().rename(columns={"index": "eid"})
                s = edict["meaning"].split("-")
                data_dict[s[0][0] + s[1][0]] = {"edge": edge_tmp[["src_idx", "dst_idx", "eid"]], "src": edict["start"], "dst": edict["end"], "etype":int(etype)}
            
            # metapath-based neighbor graph
            mp_data_dict = {}  # graph data dict
            mp_node_dict = {}  # node data dict
            mp_edge_dict = {}  # edge data dict
            self.mp_midtype_dict = {}  # the node and edge types along each metapath
            
            # join from the start node to the end node in dataframe according to each meta-path
            for metapath in self.metapath_list:
                mp_ntype_list = []  # node type list along the metapath
                mp_etype_list = []  # edge type list along the metapath
                # name the mid nodes in each meta-path with "mid_idx{a}", where a represent the number in the meta-path.
                for i in range(len(metapath)):
                    if i == 0:
                        mpdf = data_dict[metapath[i]]["edge"].rename(columns={"dst_idx":"mid_idx{}".format(i), "eid": "eid{}".format(i)})
                        mp_name = metapath[i]
                        start_type = data_dict[metapath[i]]["src"]
                        mp_ntype_list.append(int(data_dict[metapath[i]]["dst"]))
                        mp_etype_list.append(data_dict[metapath[i]]["etype"])
                    else:
                        if i == len(metapath)-1:
                            tmpdf = data_dict[metapath[i]]["edge"].rename(columns={"src_idx":"mid_idx{}".format(i-1), "eid": "eid{}".format(i)})
                            end_type = data_dict[metapath[i]]["dst"]
                        else:
                            tmpdf = data_dict[metapath[i]]["edge"].rename(columns={"src_idx":"mid_idx{}".format(i-1), "dst_idx":"mid_idx{}".format(i), "eid": "eid{}".format(i)})
                            mp_ntype_list.append(int(data_dict[metapath[i]]["dst"]))
                        mpdf = pd.merge(mpdf, tmpdf, on=["mid_idx{}".format(i-1)], how="inner")
                        mp_etype_list.append(data_dict[metapath[i]]["etype"])
                        mp_name += metapath[i][-1]  # name the meta-path with the concatenation of initials for each node type
                e_tuple = (self.node_info[start_type], mp_name, self.node_info[end_type])
                # sampling path instances
                if self.sampling:
                    mpdf = neighbor_sampling(mpdf, self.sampling)
                mp_data_dict[e_tuple] = (torch.LongTensor(mpdf.src_idx.values), torch.LongTensor(mpdf.dst_idx.values))
                mp_node_list = sorted([col for col in mpdf.columns if col.startswith("mid_idx")])
                mp_edge_list = sorted([col for col in mpdf.columns if col.startswith("eid")])
                mp_node_dict[e_tuple] = torch.from_numpy(mpdf[mp_node_list].values).to(torch.long)
                mp_edge_dict[e_tuple] = torch.from_numpy(mpdf[mp_edge_list].values).to(torch.long)
                self.mp_midtype_dict[mp_name] = [torch.LongTensor(mp_ntype_list), torch.LongTensor(mp_etype_list)]
                print("metapath: {}, sampling: {}, path instances: {}, intermediate node types: {}, intermadiate edge types: {}".format(mp_name, self.sampling, mpdf.shape[0], len(mp_node_list), len(mp_edge_list)))
            # establish the metapath-based neighbor graph
            self.g = dgl.heterograph(mp_data_dict, num_nodes_dict={self.node_info[self.target]: self.num_nodes_dict[self.node_info[self.target]]})
            # record the intermediate node and edge ids
            self.g.edata["hn"] = mp_node_dict
            self.g.edata["he"] = mp_edge_dict

            # train test split
            traindf = pd.read_csv(osp.join(self.raw_dir, "label.dat"), names=["idx", "name", "ntype", "label"], sep="\t")
            train_nodes, valid_nodes = train_test_split(traindf["idx"], test_size=0.2, random_state=42)
            self.valid_nodes = torch.from_numpy(valid_nodes.values).to(torch.long)
            self.train_nodes = torch.from_numpy(train_nodes.values).to(torch.long)
            testdf = pd.read_csv(osp.join(self.raw_dir, "label.dat.test"), names=["idx", "name", "ntype", "label"], sep="\t")
            self.test_nodes = torch.from_numpy(testdf.idx.values).to(torch.long)
            print("train set size:{}, valid set size:{}, test set size:{}".format(len(self.train_nodes), len(self.valid_nodes), len(self.test_nodes)))

            # read label
            df = pd.concat([traindf, testdf], axis=0)
            df = df.sort_values(by=["idx"], ascending=True)
            target_nodes_num = self.num_nodes_dict[self.node_info[self.target]]
            if df.shape[0] != target_nodes_num:
                df1 = pd.DataFrame(range(target_nodes_num), columns=["idx"])
                df = pd.merge(df1, df, on=["idx"], how="left")
                df["label"] = df["label"].fillna(4)
            self.label = torch.from_numpy(df.label.values).to(torch.long)

            # save graph
            self.save(processed_file_graph, processed_file_info)

    def save(self, processed_file_graph, processed_file_info):
        """
        Description
        -----------
        Save the graph and other info.

        Parameters
        ----------
        processed_file_graph: str, the saved graph file.
        processed_file_info: str, the saved info file.
        """
        info = {"train_idx": self.train_nodes,
                "valid_idx": self.valid_nodes,
                "test_idx": self.test_nodes,
                "label": self.label,
                "nfeat": self.nfeat,
                "efeat": self.efeat,
                "mp_midtype": self.mp_midtype_dict
                }
        save_graphs(processed_file_graph, self.g)
        torch.save(info, processed_file_info)

    def load(self, processed_file_graph, processed_file_info):
        """
        Description
        -----------
        Load the graph from saved file.

        Parameters
        ----------
        processed_file_graph: str, the saved graph file.
        processed_file_info: str, the saved info file.
        """
        self.g = load_graphs(processed_file_graph)[0][0]
        info = torch.load(processed_file_info)
        self.train_nodes = info["train_idx"]
        self.valid_nodes = info["valid_idx"]
        self.test_nodes = info["test_idx"]
        self.label = info["label"]
        self.nfeat = info["nfeat"]
        self.efeat = info["efeat"]
        self.mp_midtype_dict = info["mp_midtype"]

    def dim_dict(self):
        """
        Description
        -----------
        Get the initial feature dimension of each type of nodes and edges in the graph.

        Returns
        -------
        result: dict[str, int], the key is the node or edge name and the value is its corresponding dimension.
        """
        result = {}
        for k, v in self.nfeat.items():
            result[k] = v.shape[1]
        for k, v in self.efeat.items():
            result[k] = v.shape[1]

        return result

    def ntype_dict(self):
        """
        Description
        -----------
        Get the index and name of each node type.

        Returns
        -------
        dict[int, str], the key is the node type index and the value is the node type name.
        """
        return {int(k): v for k, v in self.node_info.items()}
        
    def etype_dict(self):
        """
        Description
        -----------
        Get the index and name of each edge type.

        Returns
        -------
        dict[int, str], the key is the edge type index and the value is the edge type name.
        """
        result = {}
        for k, v in self.edge_info.items():
            s = v["meaning"].split("-")
            result[int(k)] = s[0][0] + s[1][0]

        return result

    def mp_info(self, out_dim):
        """
        Description
        -----------
        Get the intermediate node and edge index along each metapath and calculate the dimension of the concatenated vectors.

        Returns
        -------
        dict[str, dict[str, any]], the key is the metapath and the value is a dict which contains "type" and "dim".
        """
        result = {}
        ndict = self.ntype_dict()
        edict = self.etype_dict()
        dimdict = self.dim_dict()
        for k, v in self.mp_midtype_dict.items():
            res = {}
            res["type"] = v
            res["dim"] = 0
            for ntype in v[0]:
                if ndict[ntype.item()] in dimdict:
                    res["dim"] += out_dim
            for etype in v[1]:
                if edict[etype.item()] in dimdict:
                    res["dim"] += out_dim
            result[k] = res

        return result


if __name__ == "__main__":
    filepath = osp.join(osp.dirname(__file__), "data/DBLP/")
    metapath_list = [["ap", "pa"], ["ap", "pv", "vp", "pa"], ["ap", "pt", "tp", "pa"]]
    dataset = DBLPMPGraph(root=filepath, metapath_list=metapath_list, sampling=20)