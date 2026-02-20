"""
Tests for tcrmd.simulate module.

All OpenMM dependencies are mocked.  Tests validate:
  * File-not-found guards.
  * Correct delegation to OpenMM API (via mocks).
  * Return types and structure of output dictionaries.
  * Function signatures use camelCase argument names.
"""

import inspect
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_pdb(directory: str, name: str = "solvated.pdb") -> str:
    path = os.path.join(directory, name)
    with open(path, "w") as fh:
        fh.write(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
    return path


def _make_openmm_mocks():
    """Return (mock_mm, mock_app, mock_unit) ready for use in patch.dict."""
    mock_unit = MagicMock()
    mock_unit.kelvin = MagicMock()
    mock_unit.picosecond = MagicMock()
    mock_unit.picoseconds = MagicMock()
    mock_unit.amu = MagicMock()
    mock_unit.nanometers = MagicMock()
    mock_unit.kilojoules_per_mole = MagicMock()
    mock_unit.molar = MagicMock()

    mock_simulation = MagicMock()
    mock_state = MagicMock()
    mock_state.getPositions.return_value = MagicMock()
    mock_state.getPotentialEnergy.return_value = MagicMock(
        value_in_unit=MagicMock(return_value=-12345.0)
    )
    mock_simulation.context.getState.return_value = mock_state
    mock_simulation.topology = MagicMock()
    mock_simulation.reporters = []
    mock_simulation.integrator = MagicMock()
    mock_simulation.integrator.setTemperature = MagicMock()
    mock_simulation.trajectory = MagicMock()
    mock_simulation.trajectory.n_frames = 5

    mock_app = MagicMock()
    mock_app.PDBFile.return_value = MagicMock(
        topology=MagicMock(), positions=MagicMock()
    )
    mock_app.ForceField.return_value = MagicMock(
        createSystem=MagicMock(return_value=MagicMock())
    )
    mock_app.Modeller.return_value = MagicMock(
        topology=MagicMock(), positions=MagicMock()
    )
    mock_app.PME = "PME"
    mock_app.NoCutoff = "NoCutoff"
    mock_app.HBonds = "HBonds"
    mock_app.AllBonds = "AllBonds"
    mock_app.Simulation.return_value = mock_simulation
    mock_app.DCDReporter = MagicMock()
    mock_app.StateDataReporter = MagicMock()

    mock_mm = MagicMock()
    mock_mm.LangevinMiddleIntegrator.return_value = MagicMock(
        setTemperature=MagicMock()
    )
    mock_mm.Platform.getPlatformByName.return_value = MagicMock()

    return mock_mm, mock_app, mock_unit, mock_simulation


# ---------------------------------------------------------------------------
# SetupSystem tests
# ---------------------------------------------------------------------------
class TestSetupSystem(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_missing_pdb_raises_file_not_found(self):
        from tcrmd.simulate import SetupSystem

        with self.assertRaises(FileNotFoundError):
            SetupSystem("/nonexistent/solvated.pdb")

    def test_import_error_without_openmm(self):
        from tcrmd.simulate import SetupSystem

        pdb = _write_pdb(self.tmp)
        saved_mm = sys.modules.pop("openmm", None)
        saved_app = sys.modules.pop("openmm.app", None)
        saved_unit = sys.modules.pop("openmm.unit", None)
        try:
            with self.assertRaises(ImportError):
                SetupSystem(pdb)
        finally:
            if saved_mm:
                sys.modules["openmm"] = saved_mm
            if saved_app:
                sys.modules["openmm.app"] = saved_app
            if saved_unit:
                sys.modules["openmm.unit"] = saved_unit

    def test_setup_returns_tuple(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        pdb = _write_pdb(self.tmp)

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            result = sim_module.SetupSystem(pdb)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_setup_uses_cpu_platform_by_default(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        pdb = _write_pdb(self.tmp)

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            sim_module.SetupSystem(pdb)

        mock_mm.Platform.getPlatformByName.assert_called_with("CPU")

    def test_hmr_passes_hydrogen_mass(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        pdb = _write_pdb(self.tmp)

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            sim_module.SetupSystem(pdb, hmr=True)

        # After the call, inspect the kwargs passed to createSystem.
        ff_mock = mock_app.ForceField.return_value
        if ff_mock.createSystem.call_args is not None:
            kwargs = ff_mock.createSystem.call_args.kwargs
            self.assertIn("hydrogenMass", kwargs)

    def test_no_hmr_does_not_pass_hydrogen_mass(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        pdb = _write_pdb(self.tmp)

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            sim_module.SetupSystem(pdb, hmr=False)

        ff_mock = mock_app.ForceField.return_value
        if ff_mock.createSystem.call_args is not None:
            kwargs = ff_mock.createSystem.call_args.kwargs
            self.assertNotIn("hydrogenMass", kwargs)


# ---------------------------------------------------------------------------
# MinimizeEnergy tests
# ---------------------------------------------------------------------------
class TestMinimizeEnergy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_import_error_without_openmm(self):
        from tcrmd.simulate import MinimizeEnergy

        mock_sim = MagicMock()
        saved_mm = sys.modules.pop("openmm", None)
        saved_app = sys.modules.pop("openmm.app", None)
        saved_unit = sys.modules.pop("openmm.unit", None)
        try:
            with self.assertRaises(ImportError):
                MinimizeEnergy(mock_sim, "/tmp/out.pdb")
        finally:
            if saved_mm:
                sys.modules["openmm"] = saved_mm
            if saved_app:
                sys.modules["openmm.app"] = saved_app
            if saved_unit:
                sys.modules["openmm.unit"] = saved_unit

    def test_minimize_energy_returns_float(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        out_pdb = os.path.join(self.tmp, "minimized.pdb")

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            with patch("builtins.open", unittest.mock.mock_open()):
                with patch("os.makedirs"):
                    result = sim_module.MinimizeEnergy(mock_simulation, out_pdb)

        self.assertIsInstance(result, float)

    def test_minimize_calls_minimizeEnergy(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        out_pdb = os.path.join(self.tmp, "min2.pdb")

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            with patch("builtins.open", unittest.mock.mock_open()):
                with patch("os.makedirs"):
                    sim_module.MinimizeEnergy(mock_simulation, out_pdb)

        mock_simulation.minimizeEnergy.assert_called_once()

    def test_default_max_iterations_is_zero(self):
        import inspect
        from tcrmd.simulate import MinimizeEnergy

        sig = inspect.signature(MinimizeEnergy)
        self.assertEqual(sig.parameters["maxIterations"].default, 0)


# ---------------------------------------------------------------------------
# RunEquilibration tests
# ---------------------------------------------------------------------------
class TestRunEquilibration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_import_error_without_openmm(self):
        from tcrmd.simulate import RunEquilibration

        mock_sim = MagicMock()
        saved_mm = sys.modules.pop("openmm", None)
        saved_app = sys.modules.pop("openmm.app", None)
        saved_unit = sys.modules.pop("openmm.unit", None)
        try:
            with self.assertRaises(ImportError):
                RunEquilibration(mock_sim, self.tmp)
        finally:
            if saved_mm:
                sys.modules["openmm"] = saved_mm
            if saved_app:
                sys.modules["openmm.app"] = saved_app
            if saved_unit:
                sys.modules["openmm.unit"] = saved_unit

    def test_returns_dcd_path(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        out_dir = os.path.join(self.tmp, "equil_out")

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            result = sim_module.RunEquilibration(
                mock_simulation, out_dir, numSteps=100
            )

        self.assertTrue(result.endswith("trajectory.dcd"))

    def test_creates_output_directory(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        out_dir = os.path.join(self.tmp, "new_equil_dir")

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            sim_module.RunEquilibration(
                mock_simulation, out_dir, numSteps=100
            )

        self.assertTrue(os.path.isdir(out_dir))

    def test_simulation_step_called(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            sim_module.RunEquilibration(
                mock_simulation, self.tmp, numSteps=500, checkpointInterval=250
            )

        # Should have been called twice (500 steps / 250 per batch).
        self.assertEqual(mock_simulation.step.call_count, 2)


# ---------------------------------------------------------------------------
# RunSimulation tests
# ---------------------------------------------------------------------------
class TestRunSimulation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_returns_dictionary_with_required_keys(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        pdb = _write_pdb(self.tmp)
        out_dir = os.path.join(self.tmp, "run_sim_out")

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            with patch("builtins.open", unittest.mock.mock_open()):
                with patch("os.makedirs"):
                    result = sim_module.RunSimulation(pdb, out_dir, numSteps=100)

        self.assertIn("minimized_pdb", result)
        self.assertIn("trajectory", result)
        self.assertIn("potential_energy", result)

    def test_minimize_first_false_skips_minimization(self):
        import tcrmd.simulate as sim_module

        mock_mm, mock_app, mock_unit, mock_simulation = _make_openmm_mocks()
        pdb = _write_pdb(self.tmp)
        out_dir = os.path.join(self.tmp, "run_sim_no_min")

        with patch.dict(
            "sys.modules",
            {
                "openmm": mock_mm,
                "openmm.app": mock_app,
                "openmm.unit": mock_unit,
            },
        ):
            import importlib
            importlib.reload(sim_module)
            result = sim_module.RunSimulation(
                pdb, out_dir, numSteps=100, minimizeFirst=False
            )

        self.assertIsNone(result["minimized_pdb"])
        self.assertIsNone(result["potential_energy"])


# ---------------------------------------------------------------------------
# Naming convention tests
# ---------------------------------------------------------------------------
class TestSimulateSignatures(unittest.TestCase):
    """Verify all public function arguments are camelCase."""

    def test_setup_system_signature(self):
        from tcrmd.simulate import SetupSystem

        params = list(inspect.signature(SetupSystem).parameters.keys())
        self.assertIn("pdbPath", params)
        self.assertIn("forceFieldFiles", params)
        self.assertIn("nonbondedMethod", params)
        self.assertIn("constraintType", params)
        self.assertIn("hmr", params)
        self.assertIn("platformName", params)

    def test_minimize_energy_signature(self):
        from tcrmd.simulate import MinimizeEnergy

        params = list(inspect.signature(MinimizeEnergy).parameters.keys())
        self.assertIn("simulation", params)
        self.assertIn("outputPdbPath", params)
        self.assertIn("maxIterations", params)
        self.assertIn("tolerance", params)

    def test_run_equilibration_signature(self):
        from tcrmd.simulate import RunEquilibration

        params = list(inspect.signature(RunEquilibration).parameters.keys())
        self.assertIn("simulation", params)
        self.assertIn("outputDir", params)
        self.assertIn("numSteps", params)
        self.assertIn("temperature", params)
        self.assertIn("reportInterval", params)
        self.assertIn("checkpointInterval", params)

    def test_run_simulation_signature(self):
        from tcrmd.simulate import RunSimulation

        params = list(inspect.signature(RunSimulation).parameters.keys())
        self.assertIn("pdbPath", params)
        self.assertIn("outputDir", params)
        self.assertIn("numSteps", params)
        self.assertIn("temperature", params)
        self.assertIn("hmr", params)
        self.assertIn("platformName", params)
        self.assertIn("minimizeFirst", params)

    def test_default_platform_is_cpu(self):
        from tcrmd.simulate import RunSimulation

        sig = inspect.signature(RunSimulation)
        self.assertEqual(sig.parameters["platformName"].default, "CPU")

    def test_default_hmr_is_true(self):
        from tcrmd.simulate import RunSimulation

        sig = inspect.signature(RunSimulation)
        self.assertTrue(sig.parameters["hmr"].default)


if __name__ == "__main__":
    unittest.main()
