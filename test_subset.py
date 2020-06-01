import unittest
import numpy as np
import msprime
import tskit
from subset import subset


def get_msprime_mig_example(N=100, T=100, n=10):
    M = [
        [0.0, 0.0],
        [0.0, 0.0]
    ]
    population_configurations = [
        msprime.PopulationConfiguration(sample_size=n),
        msprime.PopulationConfiguration(sample_size=n)
    ]
    demographic_events = [
        msprime.MassMigration(T, source=1, dest=0, proportion=1),
        msprime.CensusEvent(time=T)
    ]
    ts = msprime.simulate(
        Ne=N,
        population_configurations=population_configurations,
        demographic_events=demographic_events,
        migration_matrix=M,
        length=2e4,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        record_migrations=True)
    return ts


class TestSubsetTables(unittest.TestCase):

    def verify_subset_equality(self, tables, subset, nodes):
        # adding one so the last element always maps to NULL (-1 -> -1)
        node_map = np.repeat(tskit.NULL, tables.nodes.num_rows+1)
        indivs = []
        pops = []
        for k, n in enumerate(nodes):
            node_map[n] = k
            ind = tables.nodes[n].individual
            pop = tables.nodes[n].population
            if ind not in indivs and ind != tskit.NULL:
                indivs.append(ind)
            if pop not in pops and pop != tskit.NULL:
                pops.append(pop)
        ind_map = np.repeat(tskit.NULL, tables.individuals.num_rows+1)
        ind_map[indivs] = np.arange(len(indivs), dtype='int32')
        pop_map = np.repeat(tskit.NULL, tables.populations.num_rows+1)
        pop_map[pops] = np.arange(len(pops), dtype='int32')
        self.assertEqual(subset.nodes.num_rows, len(nodes))
        for k, n in zip(nodes, subset.nodes):
            nn = tables.nodes[k]
            self.assertEqual(nn.time, n.time)
            self.assertEqual(nn.flags, n.flags)
            self.assertEqual(nn.metadata, n.metadata)
            self.assertEqual(ind_map[nn.individual], n.individual)
            self.assertEqual(pop_map[nn.population], n.population)
        self.assertEqual(subset.individuals.num_rows, len(indivs))
        for l, i in zip(indivs, subset.individuals):
            ii = tables.individuals[l]
            self.assertEqual(ii, i)
        self.assertEqual(subset.populations.num_rows, len(pops))
        for m, p in zip(pops, subset.populations):
            pp = tables.populations[m]
            self.assertEqual(pp, p)
        edges = [i for i, e in enumerate(tables.edges) if e.parent in nodes and e.child in nodes]
        self.assertEqual(subset.edges.num_rows, len(edges))
        for q, e in zip(edges, subset.edges):
            ee = tables.edges[q]
            self.assertEqual(ee.left, e.left)
            self.assertEqual(ee.right, e.right)
            self.assertEqual(node_map[ee.parent], e.parent)
            self.assertEqual(node_map[ee.child], e.child)
            self.assertEqual(ee.metadata, e.metadata)
        muts = []
        sites = []
        for j, m in enumerate(tables.mutations):
            if m.node in nodes:
                muts.append(j)
                if m.site not in sites:
                    sites.append(m.site)
        site_map = np.arange(tables.sites.num_rows, dtype='int32')
        site_map[sites] = np.arange(len(sites), dtype='int32')
        mutation_map = np.repeat(tskit.NULL, tables.mutations.num_rows+1)
        mutation_map[muts] = np.arange(len(muts), dtype='int32')
        self.assertEqual(subset.sites.num_rows, len(sites))
        for r, s in zip(sites, subset.sites):
            ss = tables.sites[r]
            self.assertEqual(ss, s)
        self.assertEqual(subset.mutations.num_rows, len(muts))
        for t, m in zip(muts, subset.mutations):
            mm = tables.mutations[t]
            self.assertEqual(site_map[mm.site], m.site)
            self.assertEqual(node_map[mm.node], m.node)
            self.assertEqual(mm.derived_state, m.derived_state)
            if not mm.parent == m.parent == tskit.NULL:
                self.assertEqual(node_map[mm.parent], m.parent)
            self.assertEqual(mm.metadata, m.metadata)
        migs = [i for i, mig in enumerate(tables.migrations) if mig.node in nodes]
        self.assertEqual(subset.migrations.num_rows, len(migs))
        for u, mig in zip(migs, subset.migrations):
            mmig = tables.migrations[u]
            self.assertEqual(mmig.left, mig.left)
            self.assertEqual(mmig.right, mig.right)
            self.assertEqual(node_map[mmig.node], mig.node)
            self.assertEqual(pop_map[mmig.source], mig.source)
            self.assertEqual(pop_map[mmig.dest], mig.dest)
            self.assertEqual(mmig.time, mig.time)
            self.assertEqual(mmig.metadata, mig.metadata)
        nsp = subset.provenances.num_rows
        ntp = tables.provenances.num_rows
        self.assertTrue((nsp == ntp) or (nsp == ntp+1))

    def test_mig_examples(self):
        setattr(tskit.TableCollection, 'subset', subset)
        for (N, T) in [(100, 10), (1000, 100)]:
            ts = get_msprime_mig_example(N, T)
            tables = ts.tables
            n_samples = np.random.randint(1, ts.num_nodes, 10)
            for n in n_samples:
                new = tables.copy()
                nodes = np.random.choice(np.arange(ts.num_nodes), n,
                                         replace=False)
                new.subset(nodes, record_provenance=False)
                self.verify_subset_equality(tables, new, nodes)
            # assert raises error when empty nodes
            with self.assertRaises(ValueError):
                new.subset(np.array([]))
