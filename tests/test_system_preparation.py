"""
Tests for tcrmd.system_preparation.

These tests use real dependencies (pdbfixer, openmm, propka) — no mocking.
They are designed to run inside docker/system_preparation.Dockerfile, which
installs all required packages.

A minimal 5-residue alanine PDB (tests/data/minimal.pdb) is used as the
shared test fixture so that PDBFixer, PROPKA, and OpenMM all operate on a
real structure.

Tests that require a dependency that is not installed are skipped with
``pytest.importorskip``.

Naming conventions:
    Exported functions : PascalCase
    Public arguments   : camelCase
    Internal variables : snake_case
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
_MINIMAL_PDB = str(_DATA_DIR / "minimal.pdb")


# ---------------------------------------------------------------------------
# CleanPDB
# ---------------------------------------------------------------------------
class TestCleanPDB(unittest.TestCase):
    pdbfixer = None
    openmm = None

    @classmethod
    def setUpClass(cls):
        try:
            import pdbfixer  # noqa: F401
            import openmm.app  # noqa: F401
            cls.pdbfixer = True
        except ImportError:
            cls.pdbfixer = False

    def setUp(self):
        if not self.pdbfixer:
            self.skipTest("pdbfixer/openmm not installed")
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_input_raises_file_not_found(self):
        from tcrmd.system_preparation import CleanPDB
        with self.assertRaises(FileNotFoundError):
            CleanPDB("/nonexistent/input.pdb",
                     os.path.join(self.tmp, "out.pdb"))

    def test_creates_output_file(self):
        from tcrmd.system_preparation import CleanPDB
        out = os.path.join(self.tmp, "cleaned.pdb")
        CleanPDB(_MINIMAL_PDB, out)
        self.assertTrue(os.path.isfile(out))

    def test_output_contains_atom_records(self):
        from tcrmd.system_preparation import CleanPDB
        out = os.path.join(self.tmp, "cleaned_atom.pdb")
        CleanPDB(_MINIMAL_PDB, out)
        with open(out) as fh:
            content = fh.read()
        self.assertIn("ATOM", content)

    def test_returns_absolute_path(self):
        from tcrmd.system_preparation import CleanPDB
        out = os.path.join(self.tmp, "cleaned_abspath.pdb")
        result = CleanPDB(_MINIMAL_PDB, out)
        self.assertTrue(os.path.isabs(result))

    def test_hydrogens_added_when_requested(self):
        """The cleaned structure should contain hydrogen atoms."""
        from tcrmd.system_preparation import CleanPDB
        out = os.path.join(self.tmp, "cleaned_H.pdb")
        CleanPDB(_MINIMAL_PDB, out, addMissingHydrogens=True, ph=7.4)
        with open(out) as fh:
            lines = [l for l in fh if l.startswith("ATOM") or l.startswith("HETATM")]
        atom_names = [l[12:16].strip() for l in lines]
        has_hydrogen = any(n.startswith("H") for n in atom_names)
        self.assertTrue(has_hydrogen,
                        "Expected hydrogen atoms after addMissingHydrogens=True")

    def test_skip_hydrogens_when_not_requested(self):
        """When addMissingHydrogens=False, no hydrogens should be added."""
        from tcrmd.system_preparation import CleanPDB
        out = os.path.join(self.tmp, "cleaned_noH.pdb")
        CleanPDB(_MINIMAL_PDB, out, addMissingHydrogens=False)
        with open(out) as fh:
            lines = [l for l in fh if l.startswith("ATOM")]
        atom_names = [l[12:16].strip() for l in lines]
        has_hydrogen = any(n.startswith("H") for n in atom_names)
        self.assertFalse(has_hydrogen,
                         "Expected no hydrogen atoms when addMissingHydrogens=False")

    def test_creates_nested_output_dir(self):
        from tcrmd.system_preparation import CleanPDB
        nested_out = os.path.join(self.tmp, "a", "b", "cleaned.pdb")
        CleanPDB(_MINIMAL_PDB, nested_out)
        self.assertTrue(os.path.isfile(nested_out))

    def test_ph_parameter_accepted(self):
        """Verify the ph argument is accepted without error."""
        from tcrmd.system_preparation import CleanPDB
        out = os.path.join(self.tmp, "cleaned_ph6.pdb")
        CleanPDB(_MINIMAL_PDB, out, ph=6.0)
        self.assertTrue(os.path.isfile(out))


# ---------------------------------------------------------------------------
# AssignProtonationStates
# ---------------------------------------------------------------------------
class TestAssignProtonationStates(unittest.TestCase):
    propka_available = False

    @classmethod
    def setUpClass(cls):
        try:
            import propka.run  # noqa: F401
            cls.propka_available = True
        except ImportError:
            cls.propka_available = False

    def setUp(self):
        if not self.propka_available:
            self.skipTest("propka not installed")
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _get_cleaned_pdb(self) -> str:
        """Return a PDB cleaned with PDBFixer (needed by PROPKA)."""
        from tcrmd.system_preparation import CleanPDB
        cleaned = os.path.join(self.tmp, "cleaned_for_propka.pdb")
        try:
            CleanPDB(_MINIMAL_PDB, cleaned, addMissingHydrogens=True, ph=7.4)
        except ImportError:
            self.skipTest("pdbfixer/openmm not installed (needed for PROPKA input)")
        return cleaned

    def test_missing_input_raises(self):
        from tcrmd.system_preparation import AssignProtonationStates
        with self.assertRaises(FileNotFoundError):
            AssignProtonationStates("/nonexistent/input.pdb",
                                    os.path.join(self.tmp, "out.pdb"))

    def test_creates_output_file(self):
        from tcrmd.system_preparation import AssignProtonationStates
        cleaned = self._get_cleaned_pdb()
        out = os.path.join(self.tmp, "protonated.pdb")
        AssignProtonationStates(cleaned, out, ph=7.4)
        self.assertTrue(os.path.isfile(out))

    def test_output_contains_atom_records(self):
        from tcrmd.system_preparation import AssignProtonationStates
        cleaned = self._get_cleaned_pdb()
        out = os.path.join(self.tmp, "protonated_atom.pdb")
        AssignProtonationStates(cleaned, out, ph=7.4)
        with open(out) as fh:
            self.assertIn("ATOM", fh.read())

    def test_returns_absolute_path(self):
        from tcrmd.system_preparation import AssignProtonationStates
        cleaned = self._get_cleaned_pdb()
        out = os.path.join(self.tmp, "protonated_abs.pdb")
        result = AssignProtonationStates(cleaned, out, ph=7.4)
        self.assertTrue(os.path.isabs(result))

    def test_different_ph_values_accepted(self):
        """PROPKA should run without error at different pH values."""
        from tcrmd.system_preparation import AssignProtonationStates
        cleaned = self._get_cleaned_pdb()
        for ph_val in (5.0, 7.4, 9.0):
            out = os.path.join(self.tmp, f"protonated_ph{ph_val}.pdb")
            AssignProtonationStates(cleaned, out, ph=ph_val)
            self.assertTrue(os.path.isfile(out),
                            f"No output file at pH {ph_val}")


# ---------------------------------------------------------------------------
# SolvateSystem
# ---------------------------------------------------------------------------
class TestSolvateSystem(unittest.TestCase):
    openmm_available = False

    @classmethod
    def setUpClass(cls):
        try:
            import openmm.app  # noqa: F401
            cls.openmm_available = True
        except ImportError:
            cls.openmm_available = False

    def setUp(self):
        if not self.openmm_available:
            self.skipTest("openmm not installed")
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _get_cleaned_pdb(self) -> str:
        from tcrmd.system_preparation import CleanPDB
        cleaned = os.path.join(self.tmp, "cleaned_for_solv.pdb")
        CleanPDB(_MINIMAL_PDB, cleaned, addMissingHydrogens=True, ph=7.4)
        return cleaned

    def test_missing_input_raises(self):
        from tcrmd.system_preparation import SolvateSystem
        with self.assertRaises(FileNotFoundError):
            SolvateSystem("/nonexistent/input.pdb",
                          os.path.join(self.tmp, "solvated.pdb"))

    def test_creates_solvated_pdb(self):
        from tcrmd.system_preparation import SolvateSystem
        cleaned = self._get_cleaned_pdb()
        out = os.path.join(self.tmp, "solvated.pdb")
        SolvateSystem(cleaned, out)
        self.assertTrue(os.path.isfile(out))

    def test_solvated_pdb_contains_water(self):
        """After solvation the PDB should contain water residues (HOH/WAT)."""
        from tcrmd.system_preparation import SolvateSystem
        cleaned = self._get_cleaned_pdb()
        out = os.path.join(self.tmp, "solvated_water.pdb")
        SolvateSystem(cleaned, out)
        with open(out) as fh:
            content = fh.read()
        has_water = "HOH" in content or "WAT" in content
        self.assertTrue(has_water,
                        "Solvated PDB should contain water molecules")

    def test_returns_absolute_path(self):
        from tcrmd.system_preparation import SolvateSystem
        cleaned = self._get_cleaned_pdb()
        out = os.path.join(self.tmp, "solvated_abs.pdb")
        result = SolvateSystem(cleaned, out)
        self.assertTrue(os.path.isabs(result))

    def test_custom_padding_accepted(self):
        from tcrmd.system_preparation import SolvateSystem
        cleaned = self._get_cleaned_pdb()
        out = os.path.join(self.tmp, "solvated_1nm.pdb")
        SolvateSystem(cleaned, out, padding=0.8)
        self.assertTrue(os.path.isfile(out))

    def test_custom_ionic_strength_accepted(self):
        from tcrmd.system_preparation import SolvateSystem
        cleaned = self._get_cleaned_pdb()
        out = os.path.join(self.tmp, "solvated_ionic.pdb")
        SolvateSystem(cleaned, out, ionicStrength=0.1)
        self.assertTrue(os.path.isfile(out))


# ---------------------------------------------------------------------------
# Signature / naming-convention checks (no deps needed)
# ---------------------------------------------------------------------------
class TestSystemPreparationSignatures(unittest.TestCase):
    def test_clean_pdb_signature(self):
        import inspect
        from tcrmd.system_preparation import CleanPDB
        params = list(inspect.signature(CleanPDB).parameters)
        for name in ("inputPdbPath", "outputPdbPath", "addMissingHydrogens",
                     "addMissingResidues", "removeHeterogens", "ph"):
            self.assertIn(name, params)

    def test_assign_protonation_signature(self):
        import inspect
        from tcrmd.system_preparation import AssignProtonationStates
        params = list(inspect.signature(AssignProtonationStates).parameters)
        for name in ("inputPdbPath", "outputPdbPath", "ph"):
            self.assertIn(name, params)

    def test_solvate_system_signature(self):
        import inspect
        from tcrmd.system_preparation import SolvateSystem
        params = list(inspect.signature(SolvateSystem).parameters)
        for name in ("inputPdbPath", "outputPdbPath", "padding",
                     "ionicStrength", "waterModel", "positiveIon", "negativeIon"):
            self.assertIn(name, params)

    def test_default_ph_is_7_4(self):
        import inspect
        from tcrmd.system_preparation import CleanPDB
        sig = inspect.signature(CleanPDB)
        self.assertAlmostEqual(sig.parameters["ph"].default, 7.4)

    def test_default_padding_is_1_0(self):
        import inspect
        from tcrmd.system_preparation import SolvateSystem
        sig = inspect.signature(SolvateSystem)
        self.assertAlmostEqual(sig.parameters["padding"].default, 1.0)

    def test_function_names_are_pascal_case(self):
        import tcrmd.system_preparation as sp
        for name in ("CleanPDB", "AssignProtonationStates", "SolvateSystem"):
            self.assertTrue(hasattr(sp, name),
                            f"Expected PascalCase function '{name}' not found")


if __name__ == "__main__":
    unittest.main()
