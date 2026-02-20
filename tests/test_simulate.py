"""
Tests for tcrmd.simulate.

These tests use real OpenMM — no mocking.  They are designed to run inside
docker/simulate.Dockerfile, which provides an OpenMM-enabled conda env.

The test fixtures build a minimal solvated PDB by running
SolvateSystem(CleanPDB(minimal.pdb)) so that SetupSystem, MinimizeEnergy,
and RunEquilibration operate on real molecular data.

All tests requiring OpenMM are skipped automatically when the package is
absent, so the file can still be imported in a plain environment.

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

_DATA_DIR = Path(__file__).parent / "data"
_MINIMAL_PDB = str(_DATA_DIR / "minimal.pdb")


# ---------------------------------------------------------------------------
# Shared fixture: build a solvated PDB once per test class
# ---------------------------------------------------------------------------
def _make_solvated_pdb(directory: str) -> str:
    """Return path to a solvated PDB built from the minimal test fixture."""
    from tcrmd.system_preparation import CleanPDB, SolvateSystem

    cleaned = os.path.join(directory, "cleaned.pdb")
    solvated = os.path.join(directory, "solvated.pdb")
    CleanPDB(_MINIMAL_PDB, cleaned, addMissingHydrogens=True, ph=7.4)
    # Use small padding (0.5 nm) to keep the water box tiny and fast for tests.
    SolvateSystem(cleaned, solvated, padding=0.5)
    return solvated


# ---------------------------------------------------------------------------
# SetupSystem
# ---------------------------------------------------------------------------
class TestSetupSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import openmm  # noqa: F401
            import openmm.app  # noqa: F401
            cls.openmm_available = True
        except ImportError:
            cls.openmm_available = False

        if cls.openmm_available:
            cls.tmp = tempfile.mkdtemp()
            cls.solvated_pdb = _make_solvated_pdb(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if not self.openmm_available:
            self.skipTest("openmm not installed")

    def test_missing_pdb_raises_file_not_found(self):
        from tcrmd.simulate import SetupSystem
        with self.assertRaises(FileNotFoundError):
            SetupSystem("/nonexistent/solvated.pdb")

    def test_returns_two_element_tuple(self):
        from tcrmd.simulate import SetupSystem
        result = SetupSystem(self.solvated_pdb)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_first_element_is_simulation(self):
        import openmm.app as app
        from tcrmd.simulate import SetupSystem
        simulation, _ = SetupSystem(self.solvated_pdb)
        self.assertIsInstance(simulation, app.Simulation)

    def test_second_element_is_modeller(self):
        import openmm.app as app
        from tcrmd.simulate import SetupSystem
        _, modeller = SetupSystem(self.solvated_pdb)
        self.assertIsInstance(modeller, app.Modeller)

    def test_cpu_platform_selected(self):
        import openmm as mm
        from tcrmd.simulate import SetupSystem
        simulation, _ = SetupSystem(self.solvated_pdb, platformName="CPU")
        platform_name = simulation.context.getPlatform().getName()
        self.assertEqual(platform_name, "CPU")

    def test_hmr_true_increases_timestep(self):
        """With HMR enabled the integrator timestep should be 4 fs."""
        import openmm as mm
        from openmm import unit
        from tcrmd.simulate import SetupSystem
        sim_hmr, _ = SetupSystem(self.solvated_pdb, hmr=True)
        sim_nhmr, _ = SetupSystem(self.solvated_pdb, hmr=False)
        dt_hmr = sim_hmr.integrator.getStepSize().value_in_unit(unit.picoseconds)
        dt_nhmr = sim_nhmr.integrator.getStepSize().value_in_unit(unit.picoseconds)
        self.assertGreater(dt_hmr, dt_nhmr)

    def test_custom_forcefield_files_accepted(self):
        from tcrmd.simulate import SetupSystem
        simulation, _ = SetupSystem(
            self.solvated_pdb,
            forceFieldFiles=("amber14-all.xml", "amber14/tip3pfb.xml"),
        )
        self.assertIsNotNone(simulation)


# ---------------------------------------------------------------------------
# MinimizeEnergy
# ---------------------------------------------------------------------------
class TestMinimizeEnergy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import openmm  # noqa: F401
            cls.openmm_available = True
        except ImportError:
            cls.openmm_available = False

        if cls.openmm_available:
            cls.tmp = tempfile.mkdtemp()
            cls.solvated_pdb = _make_solvated_pdb(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if not self.openmm_available:
            self.skipTest("openmm not installed")

    def _make_simulation(self):
        from tcrmd.simulate import SetupSystem
        sim, _ = SetupSystem(self.solvated_pdb, platformName="CPU")
        return sim

    def test_returns_float_energy(self):
        from tcrmd.simulate import MinimizeEnergy
        sim = self._make_simulation()
        out = os.path.join(self.tmp, "minimized.pdb")
        energy = MinimizeEnergy(sim, out, maxIterations=5)
        self.assertIsInstance(energy, float)

    def test_energy_is_negative(self):
        """Potential energy of a solvated peptide should be negative (kJ/mol)."""
        from tcrmd.simulate import MinimizeEnergy
        sim = self._make_simulation()
        out = os.path.join(self.tmp, "minimized_neg.pdb")
        energy = MinimizeEnergy(sim, out, maxIterations=5)
        self.assertLess(energy, 0.0,
                        "Potential energy should be negative after minimisation")

    def test_creates_output_pdb(self):
        from tcrmd.simulate import MinimizeEnergy
        sim = self._make_simulation()
        out = os.path.join(self.tmp, "minimized_file.pdb")
        MinimizeEnergy(sim, out, maxIterations=5)
        self.assertTrue(os.path.isfile(out))

    def test_output_contains_atom_records(self):
        from tcrmd.simulate import MinimizeEnergy
        sim = self._make_simulation()
        out = os.path.join(self.tmp, "minimized_atom.pdb")
        MinimizeEnergy(sim, out, maxIterations=5)
        with open(out) as fh:
            self.assertIn("ATOM", fh.read())


# ---------------------------------------------------------------------------
# RunEquilibration
# ---------------------------------------------------------------------------
class TestRunEquilibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import openmm  # noqa: F401
            cls.openmm_available = True
        except ImportError:
            cls.openmm_available = False

        if cls.openmm_available:
            cls.tmp = tempfile.mkdtemp()
            cls.solvated_pdb = _make_solvated_pdb(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if not self.openmm_available:
            self.skipTest("openmm not installed")

    def _make_minimized_simulation(self):
        from tcrmd.simulate import MinimizeEnergy, SetupSystem
        sim, _ = SetupSystem(self.solvated_pdb, platformName="CPU")
        MinimizeEnergy(sim, os.path.join(self.tmp, "pre_equil.pdb"), maxIterations=5)
        return sim

    def test_returns_trajectory_path(self):
        from tcrmd.simulate import RunEquilibration
        sim = self._make_minimized_simulation()
        out_dir = os.path.join(self.tmp, "equil_out")
        traj = RunEquilibration(sim, out_dir, numSteps=10, reportInterval=5,
                                checkpointInterval=10)
        self.assertTrue(traj.endswith("trajectory.dcd"))

    def test_creates_dcd_file(self):
        from tcrmd.simulate import RunEquilibration
        sim = self._make_minimized_simulation()
        out_dir = os.path.join(self.tmp, "equil_dcd")
        traj = RunEquilibration(sim, out_dir, numSteps=10, reportInterval=5,
                                checkpointInterval=10)
        self.assertTrue(os.path.isfile(traj))

    def test_creates_energy_log(self):
        from tcrmd.simulate import RunEquilibration
        sim = self._make_minimized_simulation()
        out_dir = os.path.join(self.tmp, "equil_log")
        RunEquilibration(sim, out_dir, numSteps=10, reportInterval=5,
                         checkpointInterval=10)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "equil_log.csv")))

    def test_creates_checkpoint_files(self):
        from tcrmd.simulate import RunEquilibration
        sim = self._make_minimized_simulation()
        out_dir = os.path.join(self.tmp, "equil_ckpt")
        RunEquilibration(sim, out_dir, numSteps=20, reportInterval=10,
                         checkpointInterval=10)
        ckpt_files = [f for f in os.listdir(out_dir) if f.startswith("checkpoint_")]
        self.assertGreater(len(ckpt_files), 0,
                           "At least one checkpoint file should be created")


# ---------------------------------------------------------------------------
# RunSimulation (end-to-end convenience wrapper)
# ---------------------------------------------------------------------------
class TestRunSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import openmm  # noqa: F401
            cls.openmm_available = True
        except ImportError:
            cls.openmm_available = False

        if cls.openmm_available:
            cls.tmp = tempfile.mkdtemp()
            cls.solvated_pdb = _make_solvated_pdb(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmp"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if not self.openmm_available:
            self.skipTest("openmm not installed")

    def test_returns_dict_with_required_keys(self):
        from tcrmd.simulate import RunSimulation
        out_dir = os.path.join(self.tmp, "run_sim")
        result = RunSimulation(
            self.solvated_pdb, out_dir, numSteps=10,
            platformName="CPU", minimizeFirst=True,
        )
        for key in ("minimized_pdb", "trajectory", "potential_energy"):
            self.assertIn(key, result)

    def test_minimized_pdb_is_created(self):
        from tcrmd.simulate import RunSimulation
        out_dir = os.path.join(self.tmp, "run_sim_min")
        result = RunSimulation(
            self.solvated_pdb, out_dir, numSteps=10,
            platformName="CPU", minimizeFirst=True,
        )
        self.assertTrue(os.path.isfile(result["minimized_pdb"]))

    def test_trajectory_is_created(self):
        from tcrmd.simulate import RunSimulation
        out_dir = os.path.join(self.tmp, "run_sim_traj")
        result = RunSimulation(
            self.solvated_pdb, out_dir, numSteps=10,
            platformName="CPU", minimizeFirst=True,
        )
        self.assertTrue(os.path.isfile(result["trajectory"]))

    def test_potential_energy_is_negative_float(self):
        from tcrmd.simulate import RunSimulation
        out_dir = os.path.join(self.tmp, "run_sim_energy")
        result = RunSimulation(
            self.solvated_pdb, out_dir, numSteps=10,
            platformName="CPU", minimizeFirst=True,
        )
        self.assertIsInstance(result["potential_energy"], float)
        self.assertLess(result["potential_energy"], 0.0)

    def test_skip_minimization_sets_none(self):
        from tcrmd.simulate import RunSimulation
        out_dir = os.path.join(self.tmp, "run_sim_nomin")
        result = RunSimulation(
            self.solvated_pdb, out_dir, numSteps=10,
            platformName="CPU", minimizeFirst=False,
        )
        self.assertIsNone(result["minimized_pdb"])
        self.assertIsNone(result["potential_energy"])


# ---------------------------------------------------------------------------
# Signature / naming-convention checks (no deps needed)
# ---------------------------------------------------------------------------
class TestSimulateSignatures(unittest.TestCase):
    def test_setup_system_signature(self):
        from tcrmd.simulate import SetupSystem
        params = list(inspect.signature(SetupSystem).parameters)
        for name in ("pdbPath", "forceFieldFiles", "nonbondedMethod",
                     "constraintType", "hmr", "platformName"):
            self.assertIn(name, params)

    def test_minimize_energy_signature(self):
        from tcrmd.simulate import MinimizeEnergy
        params = list(inspect.signature(MinimizeEnergy).parameters)
        for name in ("simulation", "outputPdbPath", "maxIterations", "tolerance"):
            self.assertIn(name, params)

    def test_run_equilibration_signature(self):
        from tcrmd.simulate import RunEquilibration
        params = list(inspect.signature(RunEquilibration).parameters)
        for name in ("simulation", "outputDir", "numSteps", "temperature",
                     "reportInterval", "checkpointInterval"):
            self.assertIn(name, params)

    def test_run_simulation_signature(self):
        from tcrmd.simulate import RunSimulation
        params = list(inspect.signature(RunSimulation).parameters)
        for name in ("pdbPath", "outputDir", "numSteps", "temperature",
                     "hmr", "platformName", "minimizeFirst"):
            self.assertIn(name, params)

    def test_default_platform_is_cpu(self):
        from tcrmd.simulate import RunSimulation
        sig = inspect.signature(RunSimulation)
        self.assertEqual(sig.parameters["platformName"].default, "CPU")

    def test_default_hmr_is_true(self):
        from tcrmd.simulate import RunSimulation
        sig = inspect.signature(RunSimulation)
        self.assertTrue(sig.parameters["hmr"].default)

    def test_default_max_iterations_is_zero(self):
        from tcrmd.simulate import MinimizeEnergy
        sig = inspect.signature(MinimizeEnergy)
        self.assertEqual(sig.parameters["maxIterations"].default, 0)

    def test_function_names_are_pascal_case(self):
        import tcrmd.simulate as sim_mod
        for name in ("SetupSystem", "MinimizeEnergy", "RunEquilibration",
                     "RunSimulation"):
            self.assertTrue(hasattr(sim_mod, name),
                            f"Expected PascalCase function '{name}' not found")


if __name__ == "__main__":
    unittest.main()
