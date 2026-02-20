"""
Tests for tcrmd.inference_analytics.

These tests use real MDAnalysis — no mocking.  They are designed to run inside
docker/inference_analytics.Dockerfile.

Two kinds of fixture are used:
  1. A real topology + multi-model trajectory built programmatically from the
     shared minimal.pdb, so all MDAnalysis functions run on genuine data.
  2. File-existence guard tests that work even without MDAnalysis installed
     (the guards fire before the import).

Tests requiring MDAnalysis are skipped automatically when absent.

Naming conventions:
    Exported functions : PascalCase
    Public arguments   : camelCase
    Internal variables : snake_case
"""

import inspect
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent / "data"
_MINIMAL_PDB = str(_DATA_DIR / "minimal.pdb")


# ---------------------------------------------------------------------------
# Fixtures: build topology + multi-frame trajectory using MDAnalysis Writer
# ---------------------------------------------------------------------------
def _build_universe_and_traj(tmp_dir: str):
    """
    Return ``(topology_pdb, trajectory_dcd)`` paths built in *tmp_dir*.

    An MDAnalysis Universe is created from the minimal PDB.  Five trajectory
    frames are written, each with independent small displacements drawn from a
    seeded RNG (seed == frame index) so the output is fully deterministic and
    reproducible across runs.  Each frame starts from the original coordinates
    so displacements do not accumulate.
    """
    import MDAnalysis as mda

    universe = mda.Universe(_MINIMAL_PDB)
    original_positions = universe.atoms.positions.copy()
    topo_pdb = os.path.join(tmp_dir, "topology.pdb")
    traj_dcd = os.path.join(tmp_dir, "trajectory.dcd")

    # Write topology PDB (single frame, original coordinates).
    with mda.Writer(topo_pdb, n_atoms=universe.atoms.n_atoms) as w:
        w.write(universe.atoms)

    # Write a multi-frame DCD; each frame has an independent displacement
    # of up to ±0.1 Å from the original positions (non-cumulative).
    with mda.Writer(traj_dcd, n_atoms=universe.atoms.n_atoms) as w:
        for frame_idx in range(5):
            rng = np.random.default_rng(seed=frame_idx)
            displacement = rng.uniform(
                -0.1, 0.1, size=original_positions.shape
            )
            universe.atoms.positions = original_positions + displacement
            w.write(universe.atoms)

    return topo_pdb, traj_dcd
    return topo_pdb, traj_dcd


# ---------------------------------------------------------------------------
# Shared base class so the fixture is created once per test class
# ---------------------------------------------------------------------------
class _MDAnalysisTestCase(unittest.TestCase):
    mda_available = False
    tmp = None
    topo_pdb = None
    traj_dcd = None

    @classmethod
    def setUpClass(cls):
        try:
            import MDAnalysis  # noqa: F401
            cls.mda_available = True
        except ImportError:
            cls.mda_available = False
            return
        cls.tmp = tempfile.mkdtemp()
        cls.topo_pdb, cls.traj_dcd = _build_universe_and_traj(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        if cls.tmp:
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if not self.mda_available:
            self.skipTest("MDAnalysis not installed")


# ---------------------------------------------------------------------------
# _load_universe (internal helper)
# ---------------------------------------------------------------------------
class TestLoadUniverse(unittest.TestCase):
    def test_missing_topology_raises_before_import(self):
        """File-existence guard must fire without requiring MDAnalysis."""
        from tcrmd.inference_analytics import _load_universe
        with self.assertRaises(FileNotFoundError):
            _load_universe("/nonexistent/topology.pdb", None)

    def test_missing_trajectory_raises_before_import(self):
        from tcrmd.inference_analytics import _load_universe
        with self.assertRaises(FileNotFoundError):
            _load_universe(_MINIMAL_PDB, "/nonexistent/traj.dcd")


class TestLoadUniverseMDA(_MDAnalysisTestCase):
    def test_returns_universe_with_correct_n_atoms(self):
        import MDAnalysis as mda
        from tcrmd.inference_analytics import _load_universe
        universe = _load_universe(self.topo_pdb, self.traj_dcd)
        self.assertIsInstance(universe, mda.Universe)
        self.assertGreater(universe.atoms.n_atoms, 0)

    def test_topology_only_universe_works(self):
        import MDAnalysis as mda
        from tcrmd.inference_analytics import _load_universe
        universe = _load_universe(self.topo_pdb, None)
        self.assertIsInstance(universe, mda.Universe)


# ---------------------------------------------------------------------------
# ComputeRMSF
# ---------------------------------------------------------------------------
class TestComputeRMSF(_MDAnalysisTestCase):
    def test_missing_topology_raises(self):
        from tcrmd.inference_analytics import ComputeRMSF
        with self.assertRaises(FileNotFoundError):
            ComputeRMSF("/nonexistent/topo.pdb", "/nonexistent/traj.dcd")

    def test_missing_trajectory_raises(self):
        from tcrmd.inference_analytics import ComputeRMSF
        with self.assertRaises(FileNotFoundError):
            ComputeRMSF(self.topo_pdb, "/nonexistent/traj.dcd")

    def test_returns_tuple_of_numpy_arrays(self):
        from tcrmd.inference_analytics import ComputeRMSF
        resids, rmsf = ComputeRMSF(self.topo_pdb, self.traj_dcd)
        self.assertIsInstance(resids, np.ndarray)
        self.assertIsInstance(rmsf, np.ndarray)

    def test_rmsf_length_matches_residues(self):
        from tcrmd.inference_analytics import ComputeRMSF
        resids, rmsf = ComputeRMSF(self.topo_pdb, self.traj_dcd)
        self.assertEqual(len(resids), len(rmsf))

    def test_rmsf_values_are_non_negative(self):
        from tcrmd.inference_analytics import ComputeRMSF
        _, rmsf = ComputeRMSF(self.topo_pdb, self.traj_dcd)
        self.assertTrue(np.all(rmsf >= 0.0))

    def test_empty_selection_raises_value_error(self):
        from tcrmd.inference_analytics import ComputeRMSF
        with self.assertRaises(ValueError):
            ComputeRMSF(self.topo_pdb, self.traj_dcd,
                        selection="resname FAKE_RESIDUE")

    def test_output_csv_written(self):
        from tcrmd.inference_analytics import ComputeRMSF
        out_csv = os.path.join(self.tmp, "rmsf_output.csv")
        ComputeRMSF(self.topo_pdb, self.traj_dcd, outputPath=out_csv)
        self.assertTrue(os.path.isfile(out_csv))

    def test_output_csv_has_header(self):
        from tcrmd.inference_analytics import ComputeRMSF
        out_csv = os.path.join(self.tmp, "rmsf_header.csv")
        ComputeRMSF(self.topo_pdb, self.traj_dcd, outputPath=out_csv)
        with open(out_csv) as fh:
            header = fh.readline().strip()
        self.assertIn("rmsf", header.lower())

    def test_custom_selection_accepted(self):
        """Selection 'name CA' should work without error."""
        from tcrmd.inference_analytics import ComputeRMSF
        resids, rmsf = ComputeRMSF(
            self.topo_pdb, self.traj_dcd, selection="name CA"
        )
        self.assertGreater(len(resids), 0)


# ---------------------------------------------------------------------------
# ComputeContactMap
# ---------------------------------------------------------------------------
class TestComputeContactMap(_MDAnalysisTestCase):
    def test_missing_topology_raises(self):
        from tcrmd.inference_analytics import ComputeContactMap
        with self.assertRaises(FileNotFoundError):
            ComputeContactMap(
                "/nonexistent/topo.pdb", "/nonexistent/traj.dcd",
                selectionA="name CA", selectionB="name N",
            )

    def test_returns_three_element_tuple(self):
        from tcrmd.inference_analytics import ComputeContactMap
        resids_a, resids_b, mat = ComputeContactMap(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
        )
        self.assertIsInstance(resids_a, np.ndarray)
        self.assertIsInstance(resids_b, np.ndarray)
        self.assertIsInstance(mat, np.ndarray)

    def test_matrix_shape_matches_residues(self):
        from tcrmd.inference_analytics import ComputeContactMap
        resids_a, resids_b, mat = ComputeContactMap(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
        )
        self.assertEqual(mat.shape, (len(resids_a), len(resids_b)))

    def test_matrix_values_in_zero_one_range(self):
        from tcrmd.inference_analytics import ComputeContactMap
        _, _, mat = ComputeContactMap(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
        )
        self.assertTrue(np.all(mat >= 0.0))
        self.assertTrue(np.all(mat <= 1.0))

    def test_empty_selection_a_raises(self):
        from tcrmd.inference_analytics import ComputeContactMap
        with self.assertRaises(ValueError):
            ComputeContactMap(
                self.topo_pdb, self.traj_dcd,
                selectionA="resname FAKE", selectionB="name N",
            )

    def test_output_npy_written(self):
        from tcrmd.inference_analytics import ComputeContactMap
        out_npy = os.path.join(self.tmp, "contact_map.npy")
        ComputeContactMap(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
            outputPath=out_npy,
        )
        self.assertTrue(os.path.isfile(out_npy))

    def test_output_npy_loadable(self):
        from tcrmd.inference_analytics import ComputeContactMap
        out_npy = os.path.join(self.tmp, "contact_map_load.npy")
        ComputeContactMap(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
            outputPath=out_npy,
        )
        loaded = np.load(out_npy)
        self.assertIsInstance(loaded, np.ndarray)

    def test_custom_cutoff_accepted(self):
        from tcrmd.inference_analytics import ComputeContactMap
        # Use a large cutoff so every pair is in contact.
        _, _, mat = ComputeContactMap(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
            cutoff=100.0,
        )
        self.assertTrue(np.all(mat > 0.0),
                        "With a 100 Å cutoff all residue pairs should be in contact")


# ---------------------------------------------------------------------------
# ComputeCOMDistance
# ---------------------------------------------------------------------------
class TestComputeCOMDistance(_MDAnalysisTestCase):
    def test_missing_topology_raises(self):
        from tcrmd.inference_analytics import ComputeCOMDistance
        with self.assertRaises(FileNotFoundError):
            ComputeCOMDistance(
                "/nonexistent/topo.pdb", "/nonexistent/traj.dcd",
                selectionA="name CA", selectionB="name N",
            )

    def test_returns_tuple_of_arrays(self):
        from tcrmd.inference_analytics import ComputeCOMDistance
        frames, dists = ComputeCOMDistance(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
        )
        self.assertIsInstance(frames, np.ndarray)
        self.assertIsInstance(dists, np.ndarray)

    def test_length_matches_trajectory_frames(self):
        from tcrmd.inference_analytics import ComputeCOMDistance
        frames, dists = ComputeCOMDistance(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
        )
        # We wrote 5 frames in the fixture.
        self.assertEqual(len(frames), 5)
        self.assertEqual(len(dists), 5)

    def test_distances_non_negative(self):
        from tcrmd.inference_analytics import ComputeCOMDistance
        _, dists = ComputeCOMDistance(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
        )
        self.assertTrue(np.all(dists >= 0.0))

    def test_empty_selection_raises(self):
        from tcrmd.inference_analytics import ComputeCOMDistance
        with self.assertRaises(ValueError):
            ComputeCOMDistance(
                self.topo_pdb, self.traj_dcd,
                selectionA="resname FAKE", selectionB="name N",
            )

    def test_output_csv_written(self):
        from tcrmd.inference_analytics import ComputeCOMDistance
        out_csv = os.path.join(self.tmp, "com_dist.csv")
        ComputeCOMDistance(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
            outputPath=out_csv,
        )
        self.assertTrue(os.path.isfile(out_csv))

    def test_output_csv_has_header(self):
        from tcrmd.inference_analytics import ComputeCOMDistance
        out_csv = os.path.join(self.tmp, "com_header.csv")
        ComputeCOMDistance(
            self.topo_pdb, self.traj_dcd,
            selectionA="name CA", selectionB="name N",
            outputPath=out_csv,
        )
        with open(out_csv) as fh:
            header = fh.readline().strip()
        self.assertIn("com_distance", header.lower())


# ---------------------------------------------------------------------------
# ComputeHydrogenBondMap
# ---------------------------------------------------------------------------
class TestComputeHydrogenBondMap(_MDAnalysisTestCase):
    def test_missing_topology_raises(self):
        """File-guard fires before MDAnalysis is touched."""
        from tcrmd.inference_analytics import ComputeHydrogenBondMap
        with self.assertRaises(FileNotFoundError):
            ComputeHydrogenBondMap(
                "/nonexistent/topo.pdb", "/nonexistent/traj.dcd"
            )

    def test_returns_numpy_matrix(self):
        from tcrmd.inference_analytics import ComputeHydrogenBondMap
        mat = ComputeHydrogenBondMap(
            self.topo_pdb, self.traj_dcd,
            donorSelection="protein", acceptorSelection="protein",
        )
        self.assertIsInstance(mat, np.ndarray)
        self.assertEqual(mat.ndim, 2)

    def test_persistence_values_in_zero_one_range(self):
        from tcrmd.inference_analytics import ComputeHydrogenBondMap
        mat = ComputeHydrogenBondMap(
            self.topo_pdb, self.traj_dcd,
            donorSelection="protein", acceptorSelection="protein",
        )
        self.assertTrue(np.all(mat >= 0.0))
        self.assertTrue(np.all(mat <= 1.0))

    def test_output_npy_written(self):
        from tcrmd.inference_analytics import ComputeHydrogenBondMap
        out_npy = os.path.join(self.tmp, "hbond_map.npy")
        ComputeHydrogenBondMap(
            self.topo_pdb, self.traj_dcd,
            donorSelection="protein", acceptorSelection="protein",
            outputPath=out_npy,
        )
        self.assertTrue(os.path.isfile(out_npy))


# ---------------------------------------------------------------------------
# Signature / naming-convention checks (no MDAnalysis needed)
# ---------------------------------------------------------------------------
class TestInferenceAnalyticsSignatures(unittest.TestCase):
    def test_compute_rmsf_signature(self):
        from tcrmd.inference_analytics import ComputeRMSF
        params = list(inspect.signature(ComputeRMSF).parameters)
        for name in ("topologyPath", "trajectoryPath", "selection", "outputPath"):
            self.assertIn(name, params)

    def test_compute_hbond_map_signature(self):
        from tcrmd.inference_analytics import ComputeHydrogenBondMap
        params = list(inspect.signature(ComputeHydrogenBondMap).parameters)
        for name in ("topologyPath", "trajectoryPath", "donorSelection",
                     "acceptorSelection", "distanceCutoff", "angleCutoff",
                     "outputPath"):
            self.assertIn(name, params)

    def test_compute_contact_map_signature(self):
        from tcrmd.inference_analytics import ComputeContactMap
        params = list(inspect.signature(ComputeContactMap).parameters)
        for name in ("topologyPath", "trajectoryPath", "selectionA",
                     "selectionB", "cutoff", "outputPath"):
            self.assertIn(name, params)

    def test_compute_bsa_signature(self):
        from tcrmd.inference_analytics import ComputeBuriedSurfaceArea
        params = list(inspect.signature(ComputeBuriedSurfaceArea).parameters)
        for name in ("topologyPath", "trajectoryPath", "selectionA",
                     "selectionB", "probeRadius", "outputPath"):
            self.assertIn(name, params)

    def test_compute_com_distance_signature(self):
        from tcrmd.inference_analytics import ComputeCOMDistance
        params = list(inspect.signature(ComputeCOMDistance).parameters)
        for name in ("topologyPath", "trajectoryPath", "selectionA",
                     "selectionB", "outputPath"):
            self.assertIn(name, params)

    def test_default_cutoff_is_4_5(self):
        from tcrmd.inference_analytics import ComputeContactMap
        sig = inspect.signature(ComputeContactMap)
        self.assertAlmostEqual(sig.parameters["cutoff"].default, 4.5)

    def test_default_probe_radius_is_1_4(self):
        from tcrmd.inference_analytics import ComputeBuriedSurfaceArea
        sig = inspect.signature(ComputeBuriedSurfaceArea)
        self.assertAlmostEqual(sig.parameters["probeRadius"].default, 1.4)

    def test_default_distance_cutoff_is_3_5(self):
        from tcrmd.inference_analytics import ComputeHydrogenBondMap
        sig = inspect.signature(ComputeHydrogenBondMap)
        self.assertAlmostEqual(sig.parameters["distanceCutoff"].default, 3.5)

    def test_default_angle_cutoff_is_150(self):
        from tcrmd.inference_analytics import ComputeHydrogenBondMap
        sig = inspect.signature(ComputeHydrogenBondMap)
        self.assertAlmostEqual(sig.parameters["angleCutoff"].default, 150.0)

    def test_function_names_are_pascal_case(self):
        import tcrmd.inference_analytics as ia
        for name in ("ComputeRMSF", "ComputeHydrogenBondMap",
                     "ComputeContactMap", "ComputeBuriedSurfaceArea",
                     "ComputeCOMDistance"):
            self.assertTrue(hasattr(ia, name),
                            f"Expected PascalCase function '{name}' not found")


if __name__ == "__main__":
    unittest.main()
