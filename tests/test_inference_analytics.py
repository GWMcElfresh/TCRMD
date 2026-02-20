"""
Tests for tcrmd.inference_analytics module.

MDAnalysis is not available in the test environment; all tests either:
  * Test file-existence guards (which fire before any MDAnalysis import), or
  * Patch sys.modules to provide a minimal MDAnalysis mock, then call the
    analytics functions directly (lazy imports inside function bodies pick up
    the patched modules automatically).
"""

import inspect
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_topology_pdb(directory: str, name: str = "topology.pdb") -> str:
    """Write a minimal single-atom PDB to serve as a topology / trajectory file."""
    path = os.path.join(directory, name)
    with open(path, "w") as fh:
        fh.write(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
    return path


def _make_mock_universe(n_frames: int = 3, resids_a=None, resids_b=None):
    """Return (universe_mock, atoms_a_mock, atoms_b_mock)."""
    if resids_a is None:
        resids_a = np.array([1, 2])
    if resids_b is None:
        resids_b = np.array([3, 4])

    mock_u = MagicMock()
    mock_u.trajectory.n_frames = n_frames

    # atoms_a
    atoms_a = MagicMock()
    atoms_a.__len__ = MagicMock(return_value=len(resids_a))
    atoms_a.resids = resids_a
    atoms_a.resnames = np.array(["ALA"] * len(resids_a))
    atoms_a.positions = np.zeros((len(resids_a), 3))
    a_list = []
    for rid in resids_a:
        a = MagicMock()
        a.resid = rid
        a_list.append(a)
    atoms_a.__iter__ = MagicMock(return_value=iter(a_list))
    atoms_a.center_of_mass = MagicMock(return_value=np.array([0.0, 0.0, 0.0]))

    # atoms_b
    atoms_b = MagicMock()
    atoms_b.__len__ = MagicMock(return_value=len(resids_b))
    atoms_b.resids = resids_b
    atoms_b.resnames = np.array(["GLY"] * len(resids_b))
    atoms_b.positions = np.full((len(resids_b), 3), 3.0)
    b_list = []
    for rid in resids_b:
        b = MagicMock()
        b.resid = rid
        b_list.append(b)
    atoms_b.__iter__ = MagicMock(return_value=iter(b_list))
    atoms_b.center_of_mass = MagicMock(return_value=np.array([5.0, 0.0, 0.0]))

    ts_list = [MagicMock(frame=f) for f in range(n_frames)]
    mock_u.trajectory.__iter__ = MagicMock(return_value=iter(ts_list))

    return mock_u, atoms_a, atoms_b


def _build_mda_sys_modules(universe_mock):
    """Return a sys.modules dict that injects a minimal MDAnalysis mock."""
    mock_mda = MagicMock()
    mock_mda.Universe.return_value = universe_mock

    mock_analysis = MagicMock()

    rmsf_result = MagicMock()
    rmsf_result.results.rmsf = np.array([0.5, 1.2, 0.8, 2.1])
    mock_rms = MagicMock()
    mock_rms.RMSF.return_value.run.return_value = rmsf_result
    mock_rms.RMSF.return_value.results.rmsf = rmsf_result.results.rmsf
    mock_analysis.rms = mock_rms

    mock_hb = MagicMock()
    hb_result = MagicMock()
    hb_result.results.hbonds = np.empty((0, 6))
    mock_hb.HydrogenBondAnalysis.return_value.run.return_value = hb_result
    mock_hb.HydrogenBondAnalysis.return_value.results.hbonds = hb_result.results.hbonds
    mock_analysis.hydrogenbonds = mock_hb

    mock_sasa = MagicMock()
    sasa_result = MagicMock()
    sasa_result.results.areas = [np.array([100.0])]
    mock_sasa.ShrakeRupley.return_value.run.return_value = sasa_result
    mock_sasa.ShrakeRupley.return_value.results.areas = sasa_result.results.areas

    return {
        "MDAnalysis": mock_mda,
        "MDAnalysis.analysis": mock_analysis,
        "MDAnalysis.analysis.rms": mock_rms,
        "MDAnalysis.analysis.hydrogenbonds": mock_hb,
        "MDAnalysis.analysis.solvent_accessibility": mock_sasa,
        "MDAnalysis.analysis.leaflet": MagicMock(),
    }


# ---------------------------------------------------------------------------
# _load_universe tests
# ---------------------------------------------------------------------------
class TestLoadUniverse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_missing_topology_raises(self):
        from tcrmd.inference_analytics import _load_universe

        with self.assertRaises(FileNotFoundError):
            _load_universe("/nonexistent/topology.pdb", None)

    def test_missing_trajectory_raises(self):
        from tcrmd.inference_analytics import _load_universe

        topo = _write_topology_pdb(self.tmp)
        with self.assertRaises(FileNotFoundError):
            _load_universe(topo, "/nonexistent/traj.dcd")

    def test_import_error_without_mdanalysis(self):
        from tcrmd.inference_analytics import _load_universe

        topo = _write_topology_pdb(self.tmp)
        saved = sys.modules.pop("MDAnalysis", None)
        try:
            with self.assertRaises(ImportError):
                _load_universe(topo, None)
        finally:
            if saved is not None:
                sys.modules["MDAnalysis"] = saved


# ---------------------------------------------------------------------------
# ComputeRMSF tests
# ---------------------------------------------------------------------------
class TestComputeRMSF(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_missing_topology_raises(self):
        """File check fires before any MDAnalysis import."""
        from tcrmd.inference_analytics import ComputeRMSF

        with self.assertRaises(FileNotFoundError):
            ComputeRMSF("/nonexistent/topo.pdb", "/nonexistent/traj.dcd")

    def test_missing_trajectory_raises(self):
        from tcrmd.inference_analytics import ComputeRMSF

        topo = _write_topology_pdb(self.tmp)
        with self.assertRaises(FileNotFoundError):
            ComputeRMSF(topo, "/nonexistent/traj.dcd")

    def test_returns_tuple_of_arrays(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "traj.dcd")

        mock_u, atoms_a, _ = _make_mock_universe()
        # ComputeRMSF uses a single select_atoms call.
        mock_u.select_atoms.return_value = atoms_a
        atoms_a.resids = np.array([1, 2, 3, 4])

        mda_mocks = _build_mda_sys_modules(mock_u)
        # Make RMSF() return something with .run() -> .results.rmsf
        rmsf_arr = np.array([0.5, 1.2, 0.8, 2.1])
        rmsf_obj = MagicMock()
        rmsf_obj.results.rmsf = rmsf_arr
        mda_mocks["MDAnalysis.analysis.rms"].RMSF.return_value.run.return_value = (
            rmsf_obj
        )
        mda_mocks["MDAnalysis.analysis.rms"].RMSF.return_value.results.rmsf = rmsf_arr

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            resid_arr, result_rmsf = ia.ComputeRMSF(topo, traj)

        self.assertIsInstance(resid_arr, np.ndarray)
        self.assertIsInstance(result_rmsf, np.ndarray)

    def test_empty_selection_raises(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "empty_traj.dcd")

        mock_u = MagicMock()
        empty = MagicMock()
        empty.__len__ = MagicMock(return_value=0)
        mock_u.select_atoms.return_value = empty

        mda_mocks = _build_mda_sys_modules(mock_u)

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            with self.assertRaises(ValueError):
                ia.ComputeRMSF(topo, traj, selection="resname FAKE")

    def test_output_csv_written(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "rmsf_out.dcd")
        out_csv = os.path.join(self.tmp, "rmsf.csv")

        mock_u, atoms_a, _ = _make_mock_universe()
        mock_u.select_atoms.return_value = atoms_a
        atoms_a.resids = np.array([1, 2])
        atoms_a.resnames = np.array(["ALA", "GLY"])

        rmsf_arr = np.array([0.3, 0.7])
        rmsf_obj = MagicMock()
        rmsf_obj.results.rmsf = rmsf_arr
        mda_mocks = _build_mda_sys_modules(mock_u)
        mda_mocks["MDAnalysis.analysis.rms"].RMSF.return_value.run.return_value = (
            rmsf_obj
        )
        mda_mocks["MDAnalysis.analysis.rms"].RMSF.return_value.results.rmsf = rmsf_arr

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            ia.ComputeRMSF(topo, traj, outputPath=out_csv)

        self.assertTrue(os.path.isfile(out_csv))


# ---------------------------------------------------------------------------
# ComputeContactMap tests
# ---------------------------------------------------------------------------
class TestComputeContactMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_missing_topology_raises(self):
        from tcrmd.inference_analytics import ComputeContactMap

        with self.assertRaises(FileNotFoundError):
            ComputeContactMap(
                "/nonexistent/topo.pdb",
                "/nonexistent/traj.dcd",
                selectionA="protein",
                selectionB="resname LIG",
            )

    def test_returns_three_element_tuple(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "contact_traj.dcd")

        resids_a = np.array([1, 2])
        resids_b = np.array([3, 4])
        mock_u, atoms_a, atoms_b = _make_mock_universe(
            resids_a=resids_a, resids_b=resids_b
        )

        call_count = [0]

        def _sel(s):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return atoms_a
            return atoms_b

        mock_u.select_atoms.side_effect = _sel
        ts_list = [MagicMock(frame=f) for f in range(2)]
        mock_u.trajectory.__iter__ = MagicMock(return_value=iter(ts_list))

        mda_mocks = _build_mda_sys_modules(mock_u)

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            result = ia.ComputeContactMap(
                topo, traj, selectionA="segid A", selectionB="segid B"
            )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        ids_a, ids_b, mat = result
        self.assertIsInstance(mat, np.ndarray)
        self.assertEqual(mat.shape, (len(resids_a), len(resids_b)))

    def test_empty_selection_a_raises(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "empty_contact.dcd")

        mock_u = MagicMock()
        empty = MagicMock()
        empty.__len__ = MagicMock(return_value=0)
        mock_u.select_atoms.return_value = empty

        mda_mocks = _build_mda_sys_modules(mock_u)

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            with self.assertRaises(ValueError):
                ia.ComputeContactMap(
                    topo, traj, selectionA="FAKE", selectionB="ALSO_FAKE"
                )

    def test_contact_matrix_values_in_zero_one_range(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "range_traj.dcd")

        resids_a = np.array([1])
        resids_b = np.array([2])
        mock_u, atoms_a, atoms_b = _make_mock_universe(
            n_frames=3, resids_a=resids_a, resids_b=resids_b
        )
        # atoms within 4.5 Å contact distance
        atoms_a.positions = np.array([[0.0, 0.0, 0.0]])
        atoms_b.positions = np.array([[3.0, 0.0, 0.0]])

        call_c = [0]

        def sel_fn(s):
            call_c[0] += 1
            if call_c[0] % 2 == 1:
                return atoms_a
            return atoms_b

        mock_u.select_atoms.side_effect = sel_fn
        ts_list = [MagicMock(frame=f) for f in range(3)]
        mock_u.trajectory.__iter__ = MagicMock(return_value=iter(ts_list))

        mda_mocks = _build_mda_sys_modules(mock_u)

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            _, _, mat = ia.ComputeContactMap(
                topo, traj, selectionA="segid A", selectionB="segid P"
            )

        self.assertTrue(np.all(mat >= 0.0))
        self.assertTrue(np.all(mat <= 1.0))

    def test_output_file_written_when_path_given(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "out_traj.dcd")
        out_npy = os.path.join(self.tmp, "contact_map.npy")

        resids_a = np.array([1])
        resids_b = np.array([2])
        mock_u, atoms_a, atoms_b = _make_mock_universe(
            n_frames=1, resids_a=resids_a, resids_b=resids_b
        )
        atoms_a.positions = np.array([[0.0, 0.0, 0.0]])
        atoms_b.positions = np.array([[3.0, 0.0, 0.0]])

        call_c = [0]

        def sel_fn(s):
            call_c[0] += 1
            if call_c[0] % 2 == 1:
                return atoms_a
            return atoms_b

        mock_u.select_atoms.side_effect = sel_fn
        mock_u.trajectory.__iter__ = MagicMock(
            return_value=iter([MagicMock(frame=0)])
        )

        mda_mocks = _build_mda_sys_modules(mock_u)

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            ia.ComputeContactMap(
                topo,
                traj,
                selectionA="segid A",
                selectionB="segid B",
                outputPath=out_npy,
            )

        self.assertTrue(os.path.isfile(out_npy))


# ---------------------------------------------------------------------------
# ComputeCOMDistance tests
# ---------------------------------------------------------------------------
class TestComputeCOMDistance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_missing_topology_raises(self):
        from tcrmd.inference_analytics import ComputeCOMDistance

        with self.assertRaises(FileNotFoundError):
            ComputeCOMDistance(
                "/nonexistent/topo.pdb",
                "/nonexistent/traj.dcd",
                selectionA="protein",
                selectionB="resname LIG",
            )

    def test_returns_tuple_of_arrays(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "com_traj.dcd")

        mock_u, atoms_a, atoms_b = _make_mock_universe(n_frames=3)

        call_c = [0]

        def sel_fn(s):
            call_c[0] += 1
            if call_c[0] % 2 == 1:
                return atoms_a
            return atoms_b

        mock_u.select_atoms.side_effect = sel_fn
        ts_list = [MagicMock(frame=f) for f in range(3)]
        mock_u.trajectory.__iter__ = MagicMock(return_value=iter(ts_list))

        mda_mocks = _build_mda_sys_modules(mock_u)

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            frames, dists = ia.ComputeCOMDistance(topo, traj, "segid A", "segid B")

        self.assertIsInstance(frames, np.ndarray)
        self.assertIsInstance(dists, np.ndarray)
        self.assertEqual(len(frames), 3)
        self.assertEqual(len(dists), 3)

    def test_distances_are_non_negative(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "com_nn_traj.dcd")

        mock_u, atoms_a, atoms_b = _make_mock_universe(n_frames=1)
        atoms_a.center_of_mass = MagicMock(return_value=np.array([0.0, 0.0, 0.0]))
        atoms_b.center_of_mass = MagicMock(return_value=np.array([3.0, 4.0, 0.0]))

        call_c = [0]

        def sel_fn(s):
            call_c[0] += 1
            if call_c[0] % 2 == 1:
                return atoms_a
            return atoms_b

        mock_u.select_atoms.side_effect = sel_fn
        mock_u.trajectory.__iter__ = MagicMock(
            return_value=iter([MagicMock(frame=0)])
        )

        mda_mocks = _build_mda_sys_modules(mock_u)

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            _, dists = ia.ComputeCOMDistance(topo, traj, "A", "B")

        self.assertTrue(np.all(dists >= 0))
        self.assertAlmostEqual(float(dists[0]), 5.0, places=4)

    def test_empty_selection_raises(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "empty_com.dcd")

        mock_u = MagicMock()
        empty = MagicMock()
        empty.__len__ = MagicMock(return_value=0)
        mock_u.select_atoms.return_value = empty

        mda_mocks = _build_mda_sys_modules(mock_u)

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            with self.assertRaises(ValueError):
                ia.ComputeCOMDistance(topo, traj, "FAKE", "ALSO_FAKE")

    def test_output_csv_written(self):
        topo = _write_topology_pdb(self.tmp)
        traj = _write_topology_pdb(self.tmp, "csv_traj.dcd")
        out_csv = os.path.join(self.tmp, "com_distance.csv")

        mock_u, atoms_a, atoms_b = _make_mock_universe(n_frames=1)

        call_c = [0]

        def sel_fn(s):
            call_c[0] += 1
            if call_c[0] % 2 == 1:
                return atoms_a
            return atoms_b

        mock_u.select_atoms.side_effect = sel_fn
        mock_u.trajectory.__iter__ = MagicMock(
            return_value=iter([MagicMock(frame=0)])
        )

        mda_mocks = _build_mda_sys_modules(mock_u)

        with patch.dict("sys.modules", mda_mocks):
            import importlib
            import tcrmd.inference_analytics as ia
            importlib.reload(ia)
            ia.ComputeCOMDistance(topo, traj, "A", "B", outputPath=out_csv)

        self.assertTrue(os.path.isfile(out_csv))


# ---------------------------------------------------------------------------
# Naming convention / signature tests
# ---------------------------------------------------------------------------
class TestInferenceAnalyticsSignatures(unittest.TestCase):
    def test_compute_rmsf_signature(self):
        from tcrmd.inference_analytics import ComputeRMSF

        params = list(inspect.signature(ComputeRMSF).parameters.keys())
        self.assertIn("topologyPath", params)
        self.assertIn("trajectoryPath", params)
        self.assertIn("selection", params)
        self.assertIn("outputPath", params)

    def test_compute_hbond_map_signature(self):
        from tcrmd.inference_analytics import ComputeHydrogenBondMap

        params = list(inspect.signature(ComputeHydrogenBondMap).parameters.keys())
        self.assertIn("topologyPath", params)
        self.assertIn("trajectoryPath", params)
        self.assertIn("donorSelection", params)
        self.assertIn("acceptorSelection", params)
        self.assertIn("distanceCutoff", params)
        self.assertIn("angleCutoff", params)
        self.assertIn("outputPath", params)

    def test_compute_contact_map_signature(self):
        from tcrmd.inference_analytics import ComputeContactMap

        params = list(inspect.signature(ComputeContactMap).parameters.keys())
        self.assertIn("topologyPath", params)
        self.assertIn("trajectoryPath", params)
        self.assertIn("selectionA", params)
        self.assertIn("selectionB", params)
        self.assertIn("cutoff", params)
        self.assertIn("outputPath", params)

    def test_compute_bsa_signature(self):
        from tcrmd.inference_analytics import ComputeBuriedSurfaceArea

        params = list(inspect.signature(ComputeBuriedSurfaceArea).parameters.keys())
        self.assertIn("topologyPath", params)
        self.assertIn("trajectoryPath", params)
        self.assertIn("selectionA", params)
        self.assertIn("selectionB", params)
        self.assertIn("probeRadius", params)
        self.assertIn("outputPath", params)

    def test_compute_com_distance_signature(self):
        from tcrmd.inference_analytics import ComputeCOMDistance

        params = list(inspect.signature(ComputeCOMDistance).parameters.keys())
        self.assertIn("topologyPath", params)
        self.assertIn("trajectoryPath", params)
        self.assertIn("selectionA", params)
        self.assertIn("selectionB", params)
        self.assertIn("outputPath", params)

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

    def test_functions_are_pascal_case(self):
        """Verify all exported analytics functions start with an uppercase letter."""
        import tcrmd.inference_analytics as ia

        for name in [
            "ComputeRMSF",
            "ComputeHydrogenBondMap",
            "ComputeContactMap",
            "ComputeBuriedSurfaceArea",
            "ComputeCOMDistance",
        ]:
            self.assertTrue(
                hasattr(ia, name),
                f"Expected PascalCase function '{name}' not found in module.",
            )


if __name__ == "__main__":
    unittest.main()
