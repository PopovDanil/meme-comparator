import random


def top_k_sampling(indices: list) -> int:
    random.shuffle(indices)

    to_select = 0

    while indices[to_select] == -1:
        to_select += 1

    return indices[to_select]