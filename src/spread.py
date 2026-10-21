"""Spread network simulator.

The dataset tells us HOW MANY times each post was shared, but not WHO shared it to whom.
This module simulates the "who to whom" part:

  * The number of reshares is the post's REAL share count from the dataset.
  * Resharers are real genuine accounts from the dataset (with their real follower counts).
  * Who reshares from whom is simulated: accounts with more followers are more likely to be
    reshared from (preferential attachment, a standard model of social spread).
  * Time between reshares is simulated with the same random delay for every post, so any
    difference between posts comes only from their real share counts.
"""
import networkx as nx
import numpy as np
import pandas as pd


def simulate_cascade(post: pd.Series, author: pd.Series, genuine_pool: pd.DataFrame,
                     seed: int = 7, max_nodes: int = 250) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    g.add_node(author["user_id"], username=author["username"], followers=int(author["followers_count"]),
               level=0, t=0.0, role="source")
    n_shares = int(min(post["shares"], max_nodes - 1))
    pool = genuine_pool.sample(n=min(n_shares, len(genuine_pool)), random_state=seed) if n_shares else genuine_pool.iloc[:0]

    nodes = [author["user_id"]]
    for _, acc in pool.iterrows():
        weights = np.array([g.nodes[n]["followers"] + 1 for n in nodes], dtype=float)
        parent = nodes[rng.choice(len(nodes), p=weights / weights.sum())]
        t = g.nodes[parent]["t"] + rng.exponential(2.0)  # hours
        g.add_node(acc["user_id"], username=acc["username"], followers=int(acc["followers_count"]),
                   level=g.nodes[parent]["level"] + 1, t=t, role="resharer")
        g.add_edge(parent, acc["user_id"])
        nodes.append(acc["user_id"])
    return g


def cascade_stats(g: nx.DiGraph) -> dict:
    followers = [d["followers"] for _, d in g.nodes(data=True)]
    levels = [d["level"] for _, d in g.nodes(data=True)]
    times = [d["t"] for _, d in g.nodes(data=True)]
    return {
        "reshares": g.number_of_nodes() - 1,
        "potential_reach": int(sum(followers)),
        "depth": int(max(levels)),
        "hours_to_last_share": round(float(max(times)), 1),
    }


def reach_over_time(g: nx.DiGraph, hours: float = 48, step: float = 1.0) -> pd.DataFrame:
    rows = sorted((d["t"], d["followers"]) for _, d in g.nodes(data=True))
    ts = np.arange(0, hours + step, step)
    reach = [sum(f for t0, f in rows if t0 <= t) for t in ts]
    return pd.DataFrame({"hour": ts, "reach": reach})
