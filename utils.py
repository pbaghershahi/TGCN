import numpy as np
from collections import OrderedDict
import os, random, logging, torch


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


def generate_sampled_graph(triplets, sample_size, split_size, num_rels):
    edges = sample_edge_uniform(len(triplets), sample_size)
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()[:5000]
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


def filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_o = []
    for o in range(num_entities):
        if ((target_s, target_r, o) not in triplets_to_filter) or (
                (target_s, target_r, o) == (target_s, target_r, target_o)):
            filtered_o.append(o)
    return torch.LongTensor(filtered_o)


def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_s = []
    for s in range(num_entities):
        if ((s, target_r, target_o) not in triplets_to_filter) or (
                (s, target_r, target_o) == (target_s, target_r, target_o)):
            filtered_s.append(s)
    return torch.LongTensor(filtered_s)


def perturb_o_and_get_filtered_rank(model, embedding, w, s, r, o, test_size, triplets_to_filter, logger):
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 2000 == 0:
            logger.info("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_o = filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities).cuda()
        filtered_o = filtered_o.cuda()
        target_o_idx = int((filtered_o == target_o).nonzero())
        target_s = target_s.repeat(filtered_o.size())[:, None]
        target_r = target_r.repeat(filtered_o.size())[:, None]
        filtered_o = filtered_o[:, None]
        edges = torch.cat((target_s, target_r, filtered_o), dim=1)
        scores = model.predict(embedding, edges)
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_o_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)


def perturb_s_and_get_filtered_rank(model, embedding, w, s, r, o, test_size, triplets_to_filter, logger):
    num_entities = embedding.shape[0]
    num_relations = w.shape[0] // 2
    ranks = []
    for idx in range(test_size):
        if idx % 2000 == 0:
            logger.info("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_s = filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities).cuda()
        inv_target_r = target_r + num_relations
        filtered_s = filtered_s.cuda()
        target_s_idx = int((filtered_s == target_s).nonzero())
        target_o = target_o.repeat(filtered_s.size())[:, None]
        inv_target_r = inv_target_r.repeat(filtered_s.size())[:, None]
        filtered_s = filtered_s[:, None]
        edges = torch.cat((target_o, inv_target_r, filtered_s), dim=1)
        scores = model.predict(embedding, edges)
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_s_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)


def calc_filtered_mrr(model, embedding, w, train_triplets, valid_triplets, test_triplets, logger, hits=[]):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        logger.info('Perturbing subject...')
        ranks_s = perturb_s_and_get_filtered_rank(model, embedding, w, s, r, o, test_size, triplets_to_filter, logger)
        logger.info('Perturbing object...')
        ranks_o = perturb_o_and_get_filtered_rank(model, embedding, w, s, r, o, test_size, triplets_to_filter, logger)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1

        mrr = torch.mean(1.0 / ranks.float())
        logger.info("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            logger.info("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()


def calc_mrr(model, embedding, w, train_triplets, valid_triplets, test_triplets, logger, hits=[]):
    mrr = calc_filtered_mrr(model, embedding, w, train_triplets, valid_triplets, test_triplets, logger, hits)
    return mrr
