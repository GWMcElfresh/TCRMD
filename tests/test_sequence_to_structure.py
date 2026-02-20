"""
Tests for tcrmd.sequence_to_structure.

These tests use real dependencies (numpy).  Boltz itself is only needed for
RunBoltzInference; that test is skipped when the ``boltz`` CLI is absent from
PATH so the suite can run on a plain machine as well as inside the full
Docker image built from docker/sequence_to_structure.Dockerfile.

Naming conventions:
    Exported functions : PascalCase
    Public arguments   : camelCase
    Internal variables : snake_case
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tcrmd.sequence_to_structure import (
    _CDR_REGIONS,
    _extract_loop,
    _validate_sequence,
    AlignToTemplate,
    ExtractCDRLoops,
    FormatBoltzInput,
    RunBoltzInference,
)

_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Internal helper: _validate_sequence
# ---------------------------------------------------------------------------
class TestValidateSequence(unittest.TestCase):
    def test_valid_uppercase_returned_unchanged(self):
        self.assertEqual(_validate_sequence("ACDE"), "ACDE")

    def test_lowercase_is_uppercased(self):
        self.assertEqual(_validate_sequence("acde"), "ACDE")

    def test_whitespace_stripped(self):
        self.assertEqual(_validate_sequence("  ACDE  "), "ACDE")

    def test_invalid_digit_raises(self):
        with self.assertRaises(ValueError):
            _validate_sequence("ACDE1")

    def test_invalid_X_raises(self):
        with self.assertRaises(ValueError):
            _validate_sequence("ACX")

    def test_all_standard_amino_acids_accepted(self):
        all_aa = "ACDEFGHIKLMNPQRSTVWY"
        self.assertEqual(_validate_sequence(all_aa), all_aa)


# ---------------------------------------------------------------------------
# Internal helper: _extract_loop
# ---------------------------------------------------------------------------
class TestExtractLoop(unittest.TestCase):
    def test_basic_slice(self):
        # 1-indexed inclusive → Python [start-1:end]
        self.assertEqual(_extract_loop("ABCDEFGHIJ", 2, 5), "BCDE")

    def test_clamp_end_beyond_sequence(self):
        self.assertEqual(_extract_loop("ABCDE", 3, 999), "CDE")

    def test_start_before_sequence(self):
        self.assertEqual(_extract_loop("ABCDE", 0, 3), "ABC")

    def test_full_sequence(self):
        self.assertEqual(_extract_loop("ABCDE", 1, 5), "ABCDE")

    def test_empty_when_start_beyond_sequence(self):
        self.assertEqual(_extract_loop("ABC", 50, 60), "")


# ---------------------------------------------------------------------------
# CDR region definitions (_CDR_REGIONS constant)
# ---------------------------------------------------------------------------
class TestCDRRegionsDefinition(unittest.TestCase):
    def test_alpha_and_beta_present(self):
        self.assertIn("alpha", _CDR_REGIONS)
        self.assertIn("beta", _CDR_REGIONS)

    def test_three_loops_per_chain(self):
        for chain in ("alpha", "beta"):
            self.assertEqual(len(_CDR_REGIONS[chain]), 3,
                             f"{chain} should have CDR1, CDR2, CDR3")

    def test_loop_start_before_end(self):
        for chain in ("alpha", "beta"):
            for name, (start, end) in _CDR_REGIONS[chain].items():
                self.assertLess(start, end,
                                f"{chain}/{name}: start must be < end")

    def test_cdr3_starts_after_cdr2(self):
        for chain in ("alpha", "beta"):
            cdr2_end = _CDR_REGIONS[chain]["CDR2"][1]
            cdr3_start = _CDR_REGIONS[chain]["CDR3"][0]
            self.assertGreater(cdr3_start, cdr2_end)


# ---------------------------------------------------------------------------
# ExtractCDRLoops – public API
# ---------------------------------------------------------------------------
class TestExtractCDRLoops(unittest.TestCase):
    _ALPHA = "A" * 150
    _BETA = "C" * 150

    def test_returns_three_loops_alpha(self):
        result = ExtractCDRLoops({"alpha": self._ALPHA}, chainType="alpha")
        self.assertEqual(set(result), {"CDR1", "CDR2", "CDR3"})

    def test_returns_three_loops_beta(self):
        result = ExtractCDRLoops({"beta": self._BETA}, chainType="beta")
        self.assertEqual(set(result), {"CDR1", "CDR2", "CDR3"})

    def test_cdr1_alpha_correct_residues(self):
        result = ExtractCDRLoops({"alpha": self._ALPHA}, chainType="alpha")
        expected = self._ALPHA[26:38]  # IMGT 27-38 → Python [26:38]
        self.assertEqual(result["CDR1"], expected)

    def test_cdr3_alpha_correct_residues(self):
        result = ExtractCDRLoops({"alpha": self._ALPHA}, chainType="alpha")
        expected = self._ALPHA[104:117]
        self.assertEqual(result["CDR3"], expected)

    def test_case_insensitive_chain_type(self):
        result = ExtractCDRLoops({"alpha": self._ALPHA}, chainType="Alpha")
        self.assertEqual(set(result), {"CDR1", "CDR2", "CDR3"})

    def test_invalid_chain_type_raises(self):
        with self.assertRaises(ValueError):
            ExtractCDRLoops({"gamma": "ACDE"}, chainType="gamma")

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            ExtractCDRLoops({"beta": self._BETA}, chainType="alpha")

    def test_invalid_sequence_chars_raise(self):
        with self.assertRaises(ValueError):
            ExtractCDRLoops({"alpha": "ACDE123"}, chainType="alpha")

    def test_short_sequence_clips_gracefully(self):
        result = ExtractCDRLoops({"alpha": "ACDE"}, chainType="alpha")
        # CDR1 starts at index 26, which is beyond the 4-residue sequence
        self.assertEqual(result["CDR1"], "")

    def test_all_values_are_strings(self):
        result = ExtractCDRLoops({"alpha": self._ALPHA}, chainType="alpha")
        for val in result.values():
            self.assertIsInstance(val, str)

    def test_with_real_sequence_from_fixture(self):
        """Load real TCR sequences from the shared test fixture."""
        with open(_DATA_DIR / "sequences.json") as fh:
            seq_data = json.load(fh)
        # Use just the alpha chain (first 150 chars so IMGT positions are valid).
        alpha_seq = seq_data["alpha"][:150]
        result = ExtractCDRLoops({"alpha": alpha_seq}, chainType="alpha")
        self.assertIn("CDR3", result)
        self.assertIsInstance(result["CDR3"], str)


# ---------------------------------------------------------------------------
# FormatBoltzInput – public API
# ---------------------------------------------------------------------------
class TestFormatBoltzInput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.seq_data = {
            "alpha": "A" * 150,
            "beta": "C" * 150,
            "peptide": "SIINFEKL",
        }

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_file(self):
        path = FormatBoltzInput(self.seq_data, None, self.tmp)
        self.assertTrue(os.path.isfile(path))

    def test_filename_is_boltz_input_yaml(self):
        path = FormatBoltzInput(self.seq_data, None, self.tmp)
        self.assertEqual(os.path.basename(path), "boltz_input.yaml")

    def test_output_is_valid_json(self):
        path = FormatBoltzInput(self.seq_data, None, self.tmp)
        with open(path) as fh:
            data = json.load(fh)
        self.assertIn("chains", data)

    def test_all_chain_entities_present(self):
        path = FormatBoltzInput(self.seq_data, None, self.tmp)
        with open(path) as fh:
            data = json.load(fh)
        entities = {c["entity"] for c in data["chains"]}
        self.assertEqual(entities, {"alpha", "beta", "peptide"})

    def test_sequences_uppercased(self):
        seq_lower = {"alpha": "a" * 150}
        path = FormatBoltzInput(seq_lower, None, self.tmp)
        with open(path) as fh:
            data = json.load(fh)
        alpha_chain = next(c for c in data["chains"] if c["entity"] == "alpha")
        self.assertEqual(alpha_chain["sequence"], "A" * 150)

    def test_template_included_when_provided(self):
        template_pdb = os.path.join(self.tmp, "template.pdb")
        shutil.copy(_DATA_DIR / "minimal.pdb", template_pdb)
        path = FormatBoltzInput(self.seq_data, template_pdb, self.tmp)
        with open(path) as fh:
            data = json.load(fh)
        self.assertIn("templates", data)

    def test_missing_template_raises(self):
        with self.assertRaises(FileNotFoundError):
            FormatBoltzInput(self.seq_data, "/nonexistent/template.pdb", self.tmp)

    def test_empty_sequence_data_raises(self):
        with self.assertRaises(ValueError):
            FormatBoltzInput({}, None, self.tmp)

    def test_creates_nested_output_dir(self):
        nested = os.path.join(self.tmp, "a", "b", "c")
        path = FormatBoltzInput(self.seq_data, None, nested)
        self.assertTrue(os.path.isfile(path))


# ---------------------------------------------------------------------------
# RunBoltzInference – public API
# (skipped when the boltz CLI is not installed)
# ---------------------------------------------------------------------------
@unittest.skipUnless(
    shutil.which("boltz") is not None,
    "boltz CLI not found in PATH; skipping inference test",
)
class TestRunBoltzInference(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_input_raises(self):
        with self.assertRaises(FileNotFoundError):
            RunBoltzInference(
                "/nonexistent/input.yaml",
                "/nonexistent/boltz1.ckpt",
                self.tmp,
            )

    def test_missing_checkpoint_raises(self):
        input_path = os.path.join(self.tmp, "input.yaml")
        with open(input_path, "w") as fh:
            fh.write("{}")
        with self.assertRaises(FileNotFoundError):
            RunBoltzInference(input_path, "/nonexistent/boltz1.ckpt", self.tmp)

    def test_bad_checkpoint_raises_called_process_error(self):
        """A valid input but nonexistent checkpoint triggers CalledProcessError."""
        seq_data = {"alpha": "A" * 150, "peptide": "SIINFEKL"}
        out_dir = os.path.join(self.tmp, "boltz_out")
        input_path = FormatBoltzInput(seq_data, None, self.tmp)
        fake_ckpt = os.path.join(self.tmp, "fake.ckpt")
        with open(fake_ckpt, "w") as fh:
            fh.write("")
        with self.assertRaises(subprocess.CalledProcessError):
            RunBoltzInference(input_path, fake_ckpt, out_dir)


# ---------------------------------------------------------------------------
# AlignToTemplate – public API (uses only numpy)
# ---------------------------------------------------------------------------
class TestAlignToTemplate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_pdb(self, filename: str, offset: float = 0.0) -> str:
        """Write a 5-residue CA-only PDB at known coordinates."""
        path = os.path.join(self.tmp, filename)
        with open(path, "w") as fh:
            fh.write("REMARK test\n")
            for i in range(1, 6):
                x = float(i) + offset
                y = float(i) + offset
                z = float(i) + offset
                fh.write(
                    f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            fh.write("END\n")
        return path

    def test_identical_structures_rmsd_is_zero(self):
        pred = self._write_pdb("pred.pdb", 0.0)
        tmpl = self._write_pdb("tmpl.pdb", 0.0)
        out = os.path.join(self.tmp, "aligned.pdb")
        rmsd = AlignToTemplate(pred, tmpl, out)
        self.assertAlmostEqual(rmsd, 0.0, places=4)

    def test_translated_structures_aligned_to_near_zero_rmsd(self):
        pred = self._write_pdb("pred2.pdb", 0.0)
        tmpl = self._write_pdb("tmpl2.pdb", 10.0)
        out = os.path.join(self.tmp, "aligned2.pdb")
        rmsd = AlignToTemplate(pred, tmpl, out)
        self.assertAlmostEqual(rmsd, 0.0, places=4)

    def test_output_file_created(self):
        pred = self._write_pdb("pred3.pdb")
        tmpl = self._write_pdb("tmpl3.pdb")
        out = os.path.join(self.tmp, "aligned3.pdb")
        AlignToTemplate(pred, tmpl, out)
        self.assertTrue(os.path.isfile(out))

    def test_returns_float(self):
        pred = self._write_pdb("pred4.pdb")
        tmpl = self._write_pdb("tmpl4.pdb")
        out = os.path.join(self.tmp, "aligned4.pdb")
        rmsd = AlignToTemplate(pred, tmpl, out)
        self.assertIsInstance(rmsd, float)

    def test_missing_predicted_raises(self):
        tmpl = self._write_pdb("tmpl5.pdb")
        with self.assertRaises(FileNotFoundError):
            AlignToTemplate("/nonexistent/pred.pdb", tmpl,
                            os.path.join(self.tmp, "out.pdb"))

    def test_missing_template_raises(self):
        pred = self._write_pdb("pred6.pdb")
        with self.assertRaises(FileNotFoundError):
            AlignToTemplate(pred, "/nonexistent/tmpl.pdb",
                            os.path.join(self.tmp, "out.pdb"))

    def test_no_common_residues_raises(self):
        path_a = os.path.join(self.tmp, "no_common_a.pdb")
        path_b = os.path.join(self.tmp, "no_common_b.pdb")
        with open(path_a, "w") as fh:
            fh.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
        with open(path_b, "w") as fh:
            fh.write("ATOM      1  CA  ALA A 999       0.000   0.000   0.000\n")
        with self.assertRaises(RuntimeError):
            AlignToTemplate(path_a, path_b, os.path.join(self.tmp, "out_nc.pdb"))

    def test_output_contains_atom_records(self):
        pred = self._write_pdb("pred7.pdb")
        tmpl = self._write_pdb("tmpl7.pdb")
        out = os.path.join(self.tmp, "aligned7.pdb")
        AlignToTemplate(pred, tmpl, out)
        with open(out) as fh:
            self.assertIn("ATOM", fh.read())

    def test_with_real_minimal_pdb(self):
        """Align the shared minimal.pdb against itself – RMSD should be 0."""
        minimal = str(_DATA_DIR / "minimal.pdb")
        out = os.path.join(self.tmp, "aligned_minimal.pdb")
        rmsd = AlignToTemplate(minimal, minimal, out)
        self.assertAlmostEqual(rmsd, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
