import argparse, time, torch, logging, os
from datetime import datetime
from models import LinkPredict
from utils import (
    get_main, fix_seed,
    get_mappings, setup_logger,
    generate_sampled_graph,
    build_test_graph, calc_mrr
)
from torch.optim.lr_scheduler import ExponentialLR, StepLR

def main(args):
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M')+'-'+args.dataset
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./cache', exist_ok=True)
    log_file_path = './log/'+exec_name+'.log'
    model_state_file = './cache/'+exec_name+'.pth'
    logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    logger.info(args)
    seed_value = 2411
    fix_seed(seed_value, random_lib=True, numpy_lib=True, torch_lib=True)
    # load graph data
    if args.dataset == 'fb15k-237':
        ds_dir_name = './data/FB15K-237/'
    elif args.dataset == 'wn18rr':
        ds_dir_name = './data/WN18RR/'

    names2ids, rels2ids = get_mappings(
        [ds_dir_name + 'train.txt',
         ds_dir_name + 'valid.txt',
         ds_dir_name + 'test.txt']
    )

    train_data = get_main(
        ds_dir_name,
        'train.txt',
        names2ids,
        rels2ids,
        add_inverse=False
    )['triples']

    valid_data = get_main(
        ds_dir_name,
        'valid.txt',
        names2ids,
        rels2ids,
        add_inverse=False
    )['triples']

    test_data = get_main(
        ds_dir_name,
        'test.txt',
        names2ids,
        rels2ids,
        add_inverse=False
    )['triples']

    num_nodes = len(names2ids.keys())
    num_rels = len(rels2ids.keys())

    logger.info(f'train shape: {train_data.shape}, valid shape: {valid_data.shape}, test shape: {test_data.shape}')
    logger.info(f'num entities: {num_nodes}, num relations: {num_rels}')

    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)
    test_g_edges, test_norm = build_test_graph(
        num_nodes, num_rels, train_data)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_g_edges = torch.from_numpy(test_g_edges)
    test_norm = torch.from_numpy(test_norm).float()

    model = LinkPredict(
        num_nodes,
        num_rels,
        args.d_embd,
        n_layers=args.n_layers,
        cp_decompose=args.cp,
        n_bases=args.n_bases,
        reg_param=args.reg_factor,
        dropout_rates={
            'dr_input': args.dr_input,
            'dr_hid1': args.dr_hid1,
            'dr_hid2': args.dr_hid2,
            'dr_output': args.dr_output,
            'dr_decoder': args.dr_decoder
        },
        decoder=args.decoder
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step_decay, gamma=args.lr_decay)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []
    epoch = 0
    best_mrr = 0
    best_epoch = 0
    optimizer.zero_grad()
    while True:
        model.train()
        epoch += 1
        g_edges, node_id, edge_norm, edges = \
            generate_sampled_graph(train_data, args.graph_batch_size, 0.5, num_rels)

        g_edges = torch.from_numpy(g_edges)
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_norm = torch.from_numpy(edge_norm).float()
        edges = torch.from_numpy(edges)
        if use_cuda:
            g_edges = g_edges.cuda()
            node_id = node_id.cuda()
            edge_norm = edge_norm.cuda()
            edges = edges.cuda()

        t0 = time.time()
        embed = model(g_edges, node_id, edge_norm)
        loss = model.get_loss(embed, edges)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        t2 = time.time()
        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)

        if epoch % 200 == 0:
            logger.info(f"Epoch {epoch:04d} | Loss {loss:.4f} | Best MRR {best_mrr:.4f} | Best Epoch {best_epoch:05d}")
#         if (epoch in range(20000, 20005)) or (epoch in range(40000, 40005)) or (epoch in range(50000, 50005)) or (epoch in range(60000, 60005)):
        if epoch % args.evaluate_every == 0:
            with torch.no_grad():
                model.eval()
                logger.info("start eval")
                embed = model(test_g_edges.cuda(), test_node_id.cuda(), test_norm.cuda())
                mrr = calc_mrr(model, embed, model.rel_embed, torch.LongTensor(train_data).cuda(),
                               valid_data.cuda(), test_data.cuda(), logger, hits=[1, 3, 10])
                if best_mrr < mrr:
                    best_mrr = mrr
                    best_epoch = epoch
                    logger.info(f"Epoch {epoch:04d} | Loss {loss:.4f} | Best MRR {best_mrr:.4f} | Best Epoch {best_epoch:05d}")
                if epoch >= args.n_epochs:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TGCN')
    parser.add_argument("--d-embd", type=int, default=100,
                        help="number of hidden units")
    parser.add_argument("--cp", action='store_true')
    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--dr-input", type=float, default=0.2,
                        help="input dropout probability")
    parser.add_argument("--dr-hid1", type=float, default=0.3,
                        help="first hidden dropout probability")
    parser.add_argument("--dr-hid2", type=float, default=0.2,
                        help="second hidden dropout probability")
    parser.add_argument("--dr-output", type=float, default=0.2,
                        help="output dropout probability")
    parser.add_argument("--dr-decoder", type=float, default=0.2,
                        help="decoder dropout probability")
    parser.add_argument("--reg-factor", type=float, default=0.01,
                        help="L2 regularization factor")
    parser.add_argument("--n-epochs", type=int, default=15000,
                        help="number of minimum training epochs")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.95,
                        help="learning rate decay rate")
    parser.add_argument("--lr-step-decay", type=int, default=500,
                        help="decay lr every x steps")
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--evaluate-every", type=int, default=500,
                        help="perform evaluation every n epochs")
    parser.add_argument("--decoder", type=str, default="tucker",
                        help="decoder to use (possible options: tucker, distmult)")

    args = parser.parse_args()
    main(args)
