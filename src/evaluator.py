import torch
import numpy as np

KS = [5, 10, 20]
MAX_K = max(KS)
DISCOUNT = 1.0 / np.log2(np.arange(2, MAX_K + 2))


def _get_valid_users(user_train, user_test, max_users=10000):
    """Get valid users with both train and test data."""
    valid = [u for u in user_train if u in user_test and user_test[u] and user_train[u]]
    if len(valid) > max_users:
        np.random.seed(42)
        valid = list(np.random.choice(valid, max_users, replace=False))
    return valid


def _compute_batch_metrics(topk_indices, batch_users, user_test):
    """Compute hits, DCG, IDCG for a batch."""
    batch_hits = {k: [] for k in KS}
    batch_dcg = {k: [] for k in KS}
    batch_idcg = {k: [] for k in KS}
    batch_test_lens = []

    for i, user_id in enumerate(batch_users):
        test_set = set(user_test.get(user_id, []))
        batch_test_lens.append(len(test_set))

        if not test_set:
            for k in KS:
                batch_hits[k].append(0)
                batch_dcg[k].append(0)
                batch_idcg[k].append(0)
            continue

        pred = topk_indices[i]
        for k in KS:
            pred_k = pred[:k]
            hits = sum(1 for item in pred_k if item in test_set)
            batch_hits[k].append(hits)

            relevance = np.array([1.0 if item in test_set else 0.0 for item in pred_k])
            batch_dcg[k].append(np.sum(relevance * DISCOUNT[:k]))
            batch_idcg[k].append(np.sum(DISCOUNT[:min(len(test_set), k)]))

    return batch_hits, batch_dcg, batch_idcg, batch_test_lens


def _aggregate_metrics(hit_counts, dcg_scores, idcg_scores, test_lengths):
    """Aggregate batch metrics into final NDCG, HR, Recall."""
    test_lengths = np.array(test_lengths)
    valid_mask = test_lengths > 0
    n_valid = np.sum(valid_mask)

    ndcg, hr, recall = {}, {}, {}
    for k in KS:
        hits = np.array(hit_counts[k])
        dcg = np.array(dcg_scores[k])
        idcg = np.array(idcg_scores[k])

        if n_valid > 0:
            hr[k] = np.sum(hits[valid_mask] > 0) / n_valid
            recall[k] = np.mean(hits[valid_mask] / np.maximum(test_lengths[valid_mask], 1))
            ndcg[k] = np.mean(dcg[valid_mask] / np.maximum(idcg[valid_mask], 1e-8))
        else:
            hr[k], recall[k], ndcg[k] = 0.0, 0.0, 0.0

    return ndcg, hr, recall


def _mask_positive_items(scores, batch_users, user_train, num_items, device):
    """Mask out already-interacted items."""
    for i, user_id in enumerate(batch_users):
        pos = [item - 1 for item in user_train[user_id] if 0 < item <= num_items]
        if pos:
            scores[i, torch.tensor(pos, dtype=torch.long, device=device)] = float('-inf')


def evaluate(model, dataset_info, maxlen, device, batch_size=256, user_eval=None):
    """Evaluate model. user_eval overrides which set to evaluate on (val or test)."""
    model.eval()

    user_train = dataset_info['user_train']
    user_test = user_eval if user_eval is not None else dataset_info['user_test']
    num_items = dataset_info['num_items']

    valid_users = _get_valid_users(user_train, user_test)
    if not valid_users:
        return {k: 0.0 for k in KS}, {k: 0.0 for k in KS}, {k: 0.0 for k in KS}

    all_items = torch.arange(1, num_items + 1, device=device)
    with torch.no_grad():
        item_emb = model.get_item_embeddings()[all_items]

    hit_counts = {k: [] for k in KS}
    dcg_scores = {k: [] for k in KS}
    idcg_scores = {k: [] for k in KS}
    test_lengths = []

    # Dynamic features (passed if available, model ignores if unused)
    user_train_count = dataset_info.get('user_train_count')
    user_train_age = dataset_info.get('user_train_age')

    with torch.no_grad():
        for batch_start in range(0, len(valid_users), batch_size):
            batch_users = valid_users[batch_start:batch_start + batch_size]
            batch_len = len(batch_users)

            # Build sequences
            seqs = np.zeros((batch_len, maxlen), dtype=np.int32)
            seq_count_np = np.zeros((batch_len, maxlen), dtype=np.float32) if user_train_count else None
            seq_age_np = np.zeros((batch_len, maxlen), dtype=np.float32) if user_train_age else None

            for i, uid in enumerate(batch_users):
                items = user_train[uid]
                seq_len = min(len(items), maxlen)
                seqs[i, -seq_len:] = items[-seq_len:]
                if user_train_count:
                    seq_count_np[i, -seq_len:] = user_train_count[uid][-seq_len:]
                if user_train_age:
                    seq_age_np[i, -seq_len:] = user_train_age[uid][-seq_len:]

            # Forward pass
            seq_t = torch.from_numpy(seqs).long().to(device)
            seq_count_t = torch.from_numpy(seq_count_np).to(device) if user_train_count else None
            seq_age_t = torch.from_numpy(seq_age_np).to(device) if user_train_age else None

            seq_emb = model.forward(
                seq_t, seq_count=seq_count_t, seq_age=seq_age_t
            )[:, -1, :]

            scores = torch.matmul(seq_emb, item_emb.t())

            _mask_positive_items(scores, batch_users, user_train, num_items, device)

            _, topk = torch.topk(scores, MAX_K, dim=1)
            topk = topk.cpu().numpy() + 1

            hits, dcg, idcg, lens = _compute_batch_metrics(topk, batch_users, user_test)
            for k in KS:
                hit_counts[k].extend(hits[k])
                dcg_scores[k].extend(dcg[k])
                idcg_scores[k].extend(idcg[k])
            test_lengths.extend(lens)

    return _aggregate_metrics(hit_counts, dcg_scores, idcg_scores, test_lengths)
