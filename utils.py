import numpy as np
from collections import OrderedDict
import os, random, logging, torch
from collections import defaultdict

def setup_logger(
    name,
    level=logging.DEBUG,
    stream_handler=True,
    file_handler=True,
    log_file='default.log'
    ):
    open(log_file, 'w').close()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
        )

    if stream_handler:
        sth = logging.StreamHandler()
        sth.setLevel(level)
        sth.setFormatter(formatter)
        logger.addHandler(sth)

    if file_handler:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def fix_seed(seed_value, random_lib=False, numpy_lib=False, torch_lib=False):
    if random_lib:
        random.seed(seed_value)
    if numpy_lib:
        np.random.seed(seed_value)
    if torch_lib:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)


def gen_mappings(src_file_paths):
    nodes_name, relations = [], []

    for file_path in src_file_paths:
        with open(file_path, 'r') as file:
            file_lines = file.readlines()

        for line in file_lines:
            line = line.strip()
            src, rel, dst = line.split('\t')
            nodes_name += [src, dst]
            relations.append(rel)

    unique_names, unique_rels = sorted(set(nodes_name)), sorted(set(relations))
    total_names, total_rels = len(unique_names), len(unique_rels)
    names2ids = OrderedDict(zip(unique_names, range(total_names)))
    rels2ids = OrderedDict(zip(unique_rels, range(total_rels)))

    return names2ids, rels2ids


def save_mapping(mapping, mapping_save_path):
    with open(mapping_save_path, 'w') as mapping_file:
        mapping_file.write(str(len(mapping)) + '\n')
        for _name, _id in mapping.items():
            mapping_file.write(str(_name) + '\t' + str(_id) + '\n')


def get_mappings(src_file_paths):
    names2ids, rels2ids = gen_mappings(src_file_paths)
    return names2ids, rels2ids


def get_main_data(src_path, names2ids, rels2ids, dst_path=None, add_inverse=False):
    with open(src_path, 'r') as file:
        file_lines = file.readlines()

    num_entities = len(names2ids.keys())
    num_relations = len(rels2ids.keys())
    src_nodes, rel_types, dst_nodes = [], [], []
    triples = []
    nodes = []
    for line in file_lines:
        line = line.strip()
        src, rel, dst = line.split('\t')
        triples.append([names2ids[src], rels2ids[rel], names2ids[dst]])
    triples = np.array(triples)
    print(add_inverse)
    if add_inverse:
        triples = np.vstack((triples, triples[:, [2, 1, 0]]))
        triples[triples.shape[0] // 2:, 1] += num_relations

    return {
        'total_unique_nodes': len(names2ids.keys()),
        'total_unique_rels': len(rels2ids.keys()),
        'names2ids': names2ids,
        'rels2ids': rels2ids,
        'triples': triples
    }


def get_main(data_dir, file_path, names2ids, rels2ids, add_inverse):
    data_path = os.path.join(data_dir, file_path)
    main_data = get_main_data(data_path, names2ids, rels2ids, add_inverse=add_inverse)
    return main_data


def sample_edge_uniform(n_triplets, sample_size):
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)


def generate_sampled_graph(triplets, sample_size, split_size, num_rels, n_positives=5000):
    edges = sample_edge_uniform(len(triplets), sample_size)
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()[:n_positives]
    relabeled_edges = np.vstack((relabeled_edges, relabeled_edges[:, [2, 1, 0]]))
    relabeled_edges[relabeled_edges.shape[0] // 2:, 1] += num_rels
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]
    g_edges, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g_edges, uniq_v, norm, relabeled_edges


def calc_norm(src_idxs):
    uniques, inverses, counts = np.unique(src_idxs, return_inverse=True, return_counts=True)
    freqs = np.ones_like(counts)/counts
    norms = freqs[inverses]
    norms[np.isinf(norms)] = 0
    return norms[:, None]


def build_graph_from_triplets(num_nodes, num_rels, triplets):
    # g = dgl.graph(([], []))
    # g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = np.array(list(zip(src, rel, dst)))
    # dst, src, rel = edges.transpose()
    # g.add_edges(src, dst)
    norm = calc_norm(edges[:, 2])
    return edges, norm


def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))


def get_er_vocab(data):
    er_vocab = defaultdict(list)
    # print(type(data), data)
    for i in range(data.size(0)):
        er_vocab[(data[i, 0].item(), data[i, 1].item())].append(data[i, 2].item())
    return er_vocab

def calc_mrr(model, embedding, er_vocab, data, batch_size, logger):
    hits = []
    ranks = []
    for i in range(10):
        hits.append([])
    for i in range(0, len(data), batch_size):
        batch_edges = data[i:i+batch_size]
        predictions = model.predict(embedding, batch_edges)

        for j in range(batch_edges.size(0)):
            filt = er_vocab[(batch_edges[j, 0].item(), batch_edges[j, 1].item())]
            target_value = predictions[j, batch_edges[j, 2]].item()
            predictions[j, filt] = 0.0
            predictions[j, batch_edges[j, 2]] = target_value
        sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
        sort_idxs = sort_idxs.cpu().numpy()
        for j in range(batch_edges.size(0)):
            rank = np.where(sort_idxs[j]==batch_edges[j, 2].item())[0][0]
            ranks.append(rank+1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

    hit10 = np.mean(hits[9])
    hit3 = np.mean(hits[2])
    hit1 = np.mean(hits[0])
    mrr = np.mean(1./np.array(ranks))
    logger.info('Hits @10: {0}'.format(hit10))
    logger.info('Hits @3: {0}'.format(hit3))
    logger.info('Hits @1: {0}'.format(hit1))
    logger.info('Mean rank: {0}'.format(np.mean(ranks)))
    logger.info('Mean reciprocal rank: {0}'.format(mrr))
    return mrr
