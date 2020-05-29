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


def subset(tables, nodes):
    if nodes.shape[0] == 0:
        raise ValueError("Nodes cannot be empty.")
    new = tables.copy()
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
        new.individuals.clear()
    else:
        i = tables.individuals
        new.individuals.set_columns(flags=i.flags[indiv_to_keep],
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
        new.populations.clear()
    else:
        p = tables.populations
        new.populations.set_columns(**_subset_ragged_col(p.metadata,
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
    new.nodes.set_columns(flags=n.flags[nodes],
                          population=new_pops,
                          individual=new_inds,
                          time=n.time[nodes],
                          **_subset_ragged_col(n.metadata,
                                               n.metadata_offset, nodes)
                          )
    # mapping node ids in full to subsetted table
    node_map = np.arange(tables.nodes.num_rows, dtype='int32')
    node_map[nodes] = np.arange(new.nodes.num_rows, dtype='int32')
    # subsetting migrations tables
    mig = tables.migrations
    keep_mig = np.isin(mig.node, nodes)
    if keep_mig.shape[0] == 0:
        new.migrations.clear()
    else:
        new_sources = np.array([pop_map[s] for s in mig.source[keep_mig]],
                               dtype='int32')
        new_dests = np.array([pop_map[s] for s in mig.dest[keep_mig]],
                             dtype='int32')
        new.migrations.set_columns(left=mig.left[keep_mig],
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
        new.edges.clear()
    else:
        new.edges.set_columns(left=e.left[keep_edges],
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
        new.sites.clear()
        new.mutations.clear()
    else:
        new.sites.set_columns(position=s[keep_sites],
                              **_subset_ragged_col(s.ancestral_state,
                                                   s.ancestral_state_offset,
                                                   keep_sites,
                                                   "ancestral_state"),
                              **_subset_ragged_col(s.metadata,
                                                   s.metadata_offset,
                                                   keep_sites)
                              )
        site_map = np.arange(tables.sites.num_rows)
        site_map[keep_sites] = np.arange(new.sites.num_rows)
        new.mutations.set_columns(site=site_map[old_sites],
                                  node=node_map[m.node[keep_muts]],
                                  **_subset_ragged_col(m.derived_state,
                                                       m.derived_state_offset,
                                                       keep_muts,
                                                       "derived_state"),
                                  parent=node_map[m.parent],
                                  **_subset_ragged_col(m.metadata,
                                                       m.metadata_offset,
                                                       keep_muts)
                                  )
    parameters = {"command": "subset", "nodes": nodes.tolist()}
    new.provenances.add_row(
        record=json.dumps(provenance.get_provenance_dict(parameters))
    )
    return(new)
