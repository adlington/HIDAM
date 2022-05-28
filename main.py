import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="HIDAM Training")
    # model parameters
    parser.add_argument("--hidden", type=int, default=64, help="The embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout ratio")
    parser.add_argument("--l2_norm", action="store_false", default=True, help="No L2 normalization")
    parser.add_argument("--batch_norm", action="store_false", default=True, help="No batch normalization")
    # training parameters
    parser.add_argument("--opt_name", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"], help="Optimizer to use")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epoch", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--eval_metric", type=str, default="micro_f1", choices=["micro_f1", "macro_f1", "auc", "ks"], help="Evaluation metric")
    parser.add_argument("--cuda", action="store_true", default=False, help="GPU training")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from HIDAM.models.utils.data import DBLPMPGraph
    from HIDAM.models.hidam import HIDAM
    from HIDAM.models.utils.trainer import GNNTrainer

    # set the parameters
    args = parse_args()

    # read the graph dataset
    filepath = os.path.join(os.path.dirname(__file__), "data/DBLP/")
    metapath_list = [["ap", "pa"], ["ap", "pv", "vp", "pa"], ["ap", "pt", "tp", "pa"]]
    dataset = DBLPMPGraph(root=filepath, metapath_list=metapath_list, sampling=20)
    
    # create the model
    model = HIDAM(
        in_dim_dict = dataset.dim_dict(),
        hid_dim = args.hidden,
        out_dim = dataset.label.unique().shape[0],
        metapath_info = dataset.mp_info(args.hidden),
        target = dataset.node_info[dataset.target],
        ntype_dict = dataset.ntype_dict(),
        etype_dict = dataset.etype_dict(),
        activation = F.relu,
        dropout = args.dropout,
        l2_norm = args.l2_norm,
        batch_norm = args.batch_norm
        )
    # device
    if args.cuda:
        device = torch.device("cuda")
        model = model.to(device)
    else:
        device = None
    # loss function
    criterion = nn.CrossEntropyLoss()
    # set the optimizer parameters
    opt = {"opt_name": args.opt_name, "lr": args.lr, "weight_decay": args.weight_decay}
    # set the local sampling parameters
    sampling_params = {"num_layers": 1, "neighbor_sampling": False}
    # train the model
    trainer = GNNTrainer(
        dataset = dataset,
        model = model,
        opt = opt,
        criterion = criterion,
        sampling_params = sampling_params,
        batch_size = [args.batch_size, args.batch_size]
    )
    state, evaluator = trainer.train(epochs=args.epoch, eval_metric=args.eval_metric, device=device)
    # save the model
    # torch.save(state,"hidam.pth")
