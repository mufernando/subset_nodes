import tskit
import numpy as np


def main():
    ts = tskit.load("single-locus-example.trees")
    tables = ts.tables
    nodes = np.array([0, 1, 2, 3, 8, 9, 4, 5, 14])
    subset(tables, nodes)

def _subset_array(packed, offset, nodes, unpack=tskit.unpack_bytes, pack=tskit.pack_bytes):
    unpacked = np.array(unpack(packed, offset))
    return pack(unpacked[nodes])


def subset(tables, nodes):
    new = tables.copy()
    n = tables.nodes
    old_inds = n.individual[nodes]
    keep_indivs = np.unique(old_inds)  # TODO: put in order
    keep_indivs = keep_indivs[keep_indivs != tskit.NULL]
    ind_map = {ind: i for i, ind in enumerate(keep_indivs)}
    i = tables.individuals
    print(i)
    new_metadata = _subset_array(i.metadata, i.metadata_offset, keep_indivs)
    new.individuals.set_columns(flags=i.flags[keep_indivs],
                                metadata=new_metadata[0],
                                metadata_offset=new_metadata[1]
                                )
    print(new.individuals)
"""
    new_n_metadata = _subset_array(n.metadata, n.metadata_offset, nodes)
    new_inds = np.array([ind_map.get(i, -1) for i in old_inds], dtype='int32')
    new.nodes.set_columns(
        flags=n.flags[nodes],
        population=n.population[nodes],
        individual=new_inds,
        time=n.time[nodes])  # ,
# metadata=new_n_metadata[0],
# metadata_offset=new_n_metadata[1])
    node_map = np.arange(tables.nodes.num_rows)
    node_map[nodes] = np.arange(new.nodes.num_rows)

e = tables.edges
    keep_edges = np.logical_and(np.isin(e.parent, nodes), np.isin(e.child, nodes))
    new.edges.set_columns(
        left=e.left[keep_edges],
        right=e.right[keep_edges],
        parent=node_map[e.parent[keep_edges]],
        child=node_map[e.child[keep_edges]],
        **_subset_array(e.metadata, e.metadata_offset, keep_edges))"""

main()
