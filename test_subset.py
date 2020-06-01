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
        node_map = np.repeat(tskit.NULL, tables.nodes.num_rows)
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
        ind_map = np.repeat(tskit.NULL, tables.individuals.num_rows)
        ind_map[indivs] = np.arange(len(indivs), dtype='int32')
        pop_map = np.repeat(tskit.NULL, tables.populations.num_rows)
        pop_map[pops] = np.arange(len(pops), dtype='int32')
        self.assertEqual(subset.nodes.num_rows, len(nodes))
        for k, n in zip(nodes, subset.nodes):
            nn = tables.nodes[k]
            self.assertEqual(nn.time, n.time)
            self.assertEqual(nn.flags, n.flags)
            self.assertEqual(nn.metadata, n.metadata)
            if not (n.individual == nn.individual == tskit.NULL):
                self.assertEqual(ind_map[nn.individual], n.individual)
            if not (n.population == nn.population == tskit.NULL):
                self.assertEqual(pop_map[nn.population], n.population)
        for l, i in zip(indivs, subset.individuals):
            ii = tables.individuals[l]
            self.assertEqual(ii.flags, i.flags)
            self.assertEqual(ii.location, i.location)
            self.assertEqual(ii.metadata, i.metadata)
        for m, p in zip(pops, subset.populations):
            pp = tables.populations[m]
            self.assertEqual(pp.metadata, p.metadata)
        edges = [i for i, e in enumerate(tables.edges) if e.parent in nodes and e.child in nodes]
        for q, e in zip(edges, subset.edges):
            ee = tables.edges[q]
            self.assertEqual(ee.left, e.left)
            self.assertEqual(ee.right, e.right)
            self.assertEqual(node_map[ee.parent], e.parent)
            self.assertEqual(node_map[ee.child], e.child)
            self.assertEqual(ee.metadata, e.metadata)
        mutations = []
        sites = []
        for j, m in enumerate(tables.mutations):
            if m.node in nodes:
                mutations.append(j)
                if m.site not in sites:
                    sites.append(m.site)

    def test_simple_example(self):
        for (N, T) in [(100, 10), (1000, 100)]:
            ts = get_msprime_mig_example(N, T)
            tables = ts.tables
            new = tables.copy()
            n_samples = np.random.randint(1, ts.num_nodes, 1)[0]
            nodes = np.random.choice(np.arange(ts.num_nodes), n_samples,
                                     replace=False)
            setattr(tskit.TableCollection, 'subset', subset)
            new.subset(nodes, record_provenance=False)
            self.verify_subset_equality(tables, new, nodes)
