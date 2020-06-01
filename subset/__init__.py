import tskit
import numpy as np
import json
import tskit.provenance as provenance


def _subset_ragged_col(packed, offset, subset, col="metadata"):
    if col in ["ancestral_state", "derived_state", "record", "timestamp"]:
        unpack = tskit.unpack_strings
        pack = tskit.pack_strings
    elif col == "location":
        unpack = tskit.unpack_arrays
        pack = tskit.pack_arrays
    else:
        unpack = tskit.unpack_bytes
        pack = tskit.pack_bytes
    use = np.repeat(False, offset.shape[0]-1)
    use[subset] = True
    unpacked = [x for x, y in zip(unpack(packed, offset), use) if y]
    packed, offset = pack(unpacked)
    return {col: packed, col+"_offset": offset}


def subset(self, nodes, record_provenance=True):
    if nodes.shape[0] == 0:
        raise ValueError("Nodes cannot be empty.")
    if np.max(nodes) >= self.nodes.num_rows:
        raise ValueError("One of the nodes is not in the TableCollection.")
    tables = self.copy()
    n = tables.nodes
    # figuring out which individuals to keep
    old_inds = n.individual[nodes]
    indiv_to_keep, i = np.unique(old_inds, return_index=True)
    # maintaining order in which they appeard
    indiv_to_keep = indiv_to_keep[np.argsort(i)]
    # removing -1
    indiv_to_keep = indiv_to_keep[indiv_to_keep != tskit.NULL]
    # subsetting individuals table
    if indiv_to_keep.shape[0] == 0:
        self.individuals.clear()
    else:
        i = tables.individuals
        self.individuals.set_columns(flags=i.flags[indiv_to_keep],
                                     **_subset_ragged_col(
                                            i.location,
                                            i.location_offset,
                                            indiv_to_keep, "location"
                                        ),
                                     **_subset_ragged_col(
                                            i.metadata,
                                            i.metadata_offset,
                                            indiv_to_keep
                                        )
                                     )
    # figuring out which pops to keep
    old_pops = n.population[nodes]
    pop_to_keep, j = np.unique(old_pops, return_index=True)
    pop_to_keep = pop_to_keep[np.argsort(j)]
    pop_to_keep = pop_to_keep[pop_to_keep != tskit.NULL]
    # subsetting populations table
    if pop_to_keep.shape[0] == 0:
        self.populations.clear()
    else:
        p = tables.populations
        self.populations.set_columns(**_subset_ragged_col(p.metadata,
                                                          p.metadata_offset,
                                                          pop_to_keep)
                                     )
    # mapping of ind/pop id in full tables to subset tables
    ind_map = {ind: i for i, ind in enumerate(indiv_to_keep)}
    pop_map = {pop: j for j, pop in enumerate(pop_to_keep)}
    # mapping indiv/pop for nodes to keep
    new_inds = np.array([ind_map.get(i, -1) for i in old_inds],
                        dtype='int32')
    new_pops = np.array([pop_map.get(j, -1) for j in old_pops],
                        dtype='int32')
    # subsetting nodes table
    self.nodes.set_columns(flags=n.flags[nodes],
                           population=new_pops,
                           individual=new_inds,
                           time=n.time[nodes],
                           **_subset_ragged_col(n.metadata,
                                                n.metadata_offset, nodes)
                           )
    # mapping node ids in full to subsetted table
    # making the node map +1 bc last will be mapping -1 to -1
    node_map = np.arange(tables.nodes.num_rows+1, dtype='int32')
    node_map[-1] = tskit.NULL
    node_map[nodes] = np.arange(self.nodes.num_rows, dtype='int32')
    # subsetting migrations tables
    mig = tables.migrations
    keep_mig = np.isin(mig.node, nodes)
    if keep_mig.shape[0] == 0:
        self.migrations.clear()
    else:
        new_sources = np.array([pop_map.get(s, -1) for s in mig.source[keep_mig]],
                               dtype='int32')
        new_dests = np.array([pop_map.get(d, -1) for d in mig.dest[keep_mig]],
                             dtype='int32')
        self.migrations.set_columns(left=mig.left[keep_mig],
                                    right=mig.right[keep_mig],
                                    node=node_map[mig.node[keep_mig]],
                                    source=new_sources,
                                    dest=new_dests,
                                    time=mig.time[keep_mig],
                                    **_subset_ragged_col(mig.metadata,
                                                         mig.metadata_offset,
                                                         keep_mig)
                                    )
    e = tables.edges
    # keeping edges connecting nodes
    keep_edges = np.logical_and(np.isin(e.parent, nodes),
                                np.isin(e.child, nodes))
    if keep_edges.shape[0] == 0:
        self.edges.clear()
    else:
        self.edges.set_columns(left=e.left[keep_edges],
                               right=e.right[keep_edges],
                               parent=node_map[e.parent[keep_edges]],
                               child=node_map[e.child[keep_edges]],
                               **_subset_ragged_col(e.metadata,
                                                    e.metadata_offset,
                                                    keep_edges)
                               )
    # subsetting mutation and sites tables
    m = tables.mutations
    s = tables.sites
    # only keeping muts in nodes
    keep_muts = np.isin(m.node, nodes)
    # only keeping sites of muts in nodes
    old_sites = m.site[keep_muts]
    keep_sites = np.unique(old_sites)
    if keep_sites.shape[0] == 0:
        self.sites.clear()
        self.mutations.clear()
    else:
        self.sites.set_columns(position=s.position[keep_sites],
                               **_subset_ragged_col(s.ancestral_state,
                                                    s.ancestral_state_offset,
                                                    keep_sites,
                                                    "ancestral_state"),
                               **_subset_ragged_col(s.metadata,
                                                    s.metadata_offset,
                                                    keep_sites)
                               )
        site_map = np.arange(tables.sites.num_rows, dtype='int32')
        site_map[keep_sites] = np.arange(self.sites.num_rows, dtype='int32')
        mutation_map = np.arange(tables.mutations.num_rows, dtype='int32')
        mutation_map[keep_muts] = np.arange(np.count_nonzero(keep_muts), dtype='int32')
        # adding tskit.NULL to the end to map -1 -> -1
        mutation_map = np.concatenate((mutation_map, np.array([tskit.NULL], dtype='int32')))
        self.mutations.set_columns(site=site_map[old_sites],
                                   node=node_map[m.node[keep_muts]],
                                   **_subset_ragged_col(m.derived_state,
                                                        m.derived_state_offset,
                                                        keep_muts,
                                                        "derived_state"),
                                   parent=mutation_map[m.parent[keep_muts]],
                                   **_subset_ragged_col(m.metadata,
                                                        m.metadata_offset,
                                                        keep_muts)
                                   )
    parameters = {"command": "subset", "nodes": nodes.tolist()}
    if record_provenance:
        self.provenances.add_row(
            record=json.dumps(provenance.get_provenance_dict(parameters))
        )
