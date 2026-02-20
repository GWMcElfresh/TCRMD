"""
Tests for tcrmd.sequence_to_structure module.

Heavy external dependencies (boltz, numpy for alignment) are mocked so the
unit tests can run in a plain Python environment (pytest + numpy only).
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from tcrmd.sequence_to_structure import (
    AlignToTemplate,
    ExtractCDRLoops,
    FormatBoltzInput,
    RunBoltzInference,
    _CDR_REGIONS,
    _extract_loop,
    _validate_sequence,
)


class TestValidateSequence(unittest.TestCase):
    """Tests for the internal _validate_sequence helper."""

    def test_valid_sequence_uppercase(self):
        result = _validate_sequence("ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(result, "ACDEFGHIKLMNPQRSTVWY")

    def test_valid_sequence_lowercase_converted(self):
        result = _validate_sequence("acde")
        self.assertEqual(result, "ACDE")

    def test_valid_sequence_with_whitespace(self):
        result = _validate_sequence("  ACDE  ")
        self.assertEqual(result, "ACDE")

    def test_invalid_characters_raise(self):
        with self.assertRaises(ValueError):
            _validate_sequence("ACDE1234")

    def test_invalid_character_X_raises(self):
        with self.assertRaises(ValueError):
            _validate_sequence("ACXDE")


class TestExtractLoop(unittest.TestCase):
    """Tests for the internal _extract_loop helper."""

    def test_basic_extraction(self):
        seq = "ABCDEFGHIJ"
        # 1-indexed, inclusive: positions 2-5 → "BCDE"
        self.assertEqual(_extract_loop(seq, 2, 5), "BCDE")

    def test_clamp_to_sequence_length(self):
        seq = "ABCDE"
        result = _extract_loop(seq, 3, 100)
        self.assertEqual(result, "CDE")

    def test_start_before_sequence(self):
        seq = "ABCDE"
        result = _extract_loop(seq, 0, 3)
        self.assertEqual(result, "ABC")

    def test_full_sequence(self):
        seq = "ABCDE"
        result = _extract_loop(seq, 1, 5)
        self.assertEqual(result, "ABCDE")


class TestExtractCDRLoops(unittest.TestCase):
    """Tests for ExtractCDRLoops."""

    # 150-residue dummy alpha chain sequence.
    _ALPHA_SEQ = "A" * 150
    # 150-residue dummy beta chain sequence.
    _BETA_SEQ = "C" * 150

    def _make_sequence_data(self, chain: str = "alpha") -> dict:
        return {chain: self._ALPHA_SEQ if chain == "alpha" else self._BETA_SEQ}

    def test_returns_three_cdr_loops_alpha(self):
        result = ExtractCDRLoops({"alpha": self._ALPHA_SEQ}, chainType="alpha")
        self.assertIn("CDR1", result)
        self.assertIn("CDR2", result)
        self.assertIn("CDR3", result)

    def test_returns_three_cdr_loops_beta(self):
        result = ExtractCDRLoops({"beta": self._BETA_SEQ}, chainType="beta")
        self.assertEqual(set(result.keys()), {"CDR1", "CDR2", "CDR3"})

    def test_cdr1_alpha_correct_range(self):
        # CDR1 alpha: IMGT 27-38 (1-indexed), length 12
        result = ExtractCDRLoops({"alpha": self._ALPHA_SEQ}, chainType="alpha")
        expected = self._ALPHA_SEQ[26:38]  # Python 0-indexed
        self.assertEqual(result["CDR1"], expected)

    def test_cdr3_alpha_correct_range(self):
        # CDR3 alpha: IMGT 105-117, length 13
        result = ExtractCDRLoops({"alpha": self._ALPHA_SEQ}, chainType="alpha")
        expected = self._ALPHA_SEQ[104:117]
        self.assertEqual(result["CDR3"], expected)

    def test_invalid_chain_type_raises(self):
        with self.assertRaises(ValueError):
            ExtractCDRLoops({"gamma": "ACDE"}, chainType="gamma")

    def test_missing_chain_key_raises(self):
        with self.assertRaises(KeyError):
            ExtractCDRLoops({"beta": self._BETA_SEQ}, chainType="alpha")

    def test_invalid_sequence_characters_raise(self):
        with self.assertRaises(ValueError):
            ExtractCDRLoops({"alpha": "ACDE123"}, chainType="alpha")

    def test_case_insensitive_chain_type(self):
        result = ExtractCDRLoops({"alpha": self._ALPHA_SEQ}, chainType="Alpha")
        self.assertEqual(set(result.keys()), {"CDR1", "CDR2", "CDR3"})

    def test_short_sequence_gracefully_clips(self):
        short_seq = "ACDEFGHIKLMN"  # only 12 residues
        result = ExtractCDRLoops({"alpha": short_seq}, chainType="alpha")
        # CDR1 starts at index 26, which is beyond the sequence → empty
        self.assertEqual(result["CDR1"], "")

    def test_returns_string_values(self):
        result = ExtractCDRLoops({"alpha": self._ALPHA_SEQ}, chainType="alpha")
        for key, val in result.items():
            self.assertIsInstance(val, str, f"{key} should be a string")


class TestFormatBoltzInput(unittest.TestCase):
    """Tests for FormatBoltzInput."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.sequence_data = {
            "alpha": "A" * 150,
            "beta": "C" * 150,
            "peptide": "SIINFEKL",
        }

    def test_creates_output_file(self):
        out_path = FormatBoltzInput(
            self.sequence_data, templatePdbPath=None, outputDir=self.tmp_dir
        )
        self.assertTrue(os.path.isfile(out_path))

    def test_output_filename(self):
        out_path = FormatBoltzInput(
            self.sequence_data, templatePdbPath=None, outputDir=self.tmp_dir
        )
        self.assertEqual(os.path.basename(out_path), "boltz_input.yaml")

    def test_output_is_valid_json(self):
        out_path = FormatBoltzInput(
            self.sequence_data, templatePdbPath=None, outputDir=self.tmp_dir
        )
        with open(out_path) as fh:
            data = json.load(fh)
        self.assertIn("chains", data)

    def test_all_chains_present(self):
        out_path = FormatBoltzInput(
            self.sequence_data, templatePdbPath=None, outputDir=self.tmp_dir
        )
        with open(out_path) as fh:
            data = json.load(fh)
        entities = {c["entity"] for c in data["chains"]}
        self.assertEqual(entities, {"alpha", "beta", "peptide"})

    def test_template_path_included_when_provided(self):
        # Create a dummy template PDB.
        template_pdb = os.path.join(self.tmp_dir, "template.pdb")
        with open(template_pdb, "w") as fh:
            fh.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")

        out_path = FormatBoltzInput(
            self.sequence_data,
            templatePdbPath=template_pdb,
            outputDir=self.tmp_dir,
        )
        with open(out_path) as fh:
            data = json.load(fh)
        self.assertIn("templates", data)
        self.assertEqual(len(data["templates"]), 1)

    def test_missing_template_raises(self):
        with self.assertRaises(FileNotFoundError):
            FormatBoltzInput(
                self.sequence_data,
                templatePdbPath="/nonexistent/template.pdb",
                outputDir=self.tmp_dir,
            )

    def test_empty_sequence_data_raises(self):
        with self.assertRaises(ValueError):
            FormatBoltzInput({}, templatePdbPath=None, outputDir=self.tmp_dir)

    def test_creates_output_dir_if_missing(self):
        new_dir = os.path.join(self.tmp_dir, "nested", "dir")
        out_path = FormatBoltzInput(
            self.sequence_data, templatePdbPath=None, outputDir=new_dir
        )
        self.assertTrue(os.path.isfile(out_path))

    def test_sequences_are_uppercased_in_output(self):
        data_lower = {"alpha": "a" * 150}
        out_path = FormatBoltzInput(
            data_lower, templatePdbPath=None, outputDir=self.tmp_dir
        )
        with open(out_path) as fh:
            data = json.load(fh)
        alpha_chain = next(c for c in data["chains"] if c["entity"] == "alpha")
        self.assertEqual(alpha_chain["sequence"], "A" * 150)


class TestRunBoltzInference(unittest.TestCase):
    """Tests for RunBoltzInference – subprocess call is mocked."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def _create_dummy_files(self):
        input_path = os.path.join(self.tmp_dir, "boltz_input.yaml")
        checkpoint_path = os.path.join(self.tmp_dir, "boltz1.ckpt")
        with open(input_path, "w") as fh:
            fh.write("{}")
        with open(checkpoint_path, "w") as fh:
            fh.write("")
        return input_path, checkpoint_path

    def test_missing_input_raises(self):
        with self.assertRaises(FileNotFoundError):
            RunBoltzInference(
                "/nonexistent/input.yaml",
                "/nonexistent/checkpoint.ckpt",
                self.tmp_dir,
            )

    def test_missing_checkpoint_raises(self):
        input_path = os.path.join(self.tmp_dir, "input.yaml")
        with open(input_path, "w") as fh:
            fh.write("{}")
        with self.assertRaises(FileNotFoundError):
            RunBoltzInference(
                input_path,
                "/nonexistent/checkpoint.ckpt",
                self.tmp_dir,
            )

    @patch("tcrmd.sequence_to_structure.subprocess.run")
    def test_subprocess_called_with_correct_args(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        input_path, checkpoint_path = self._create_dummy_files()
        output_dir = os.path.join(self.tmp_dir, "out")

        RunBoltzInference(input_path, checkpoint_path, output_dir)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("boltz", cmd)
        self.assertIn("predict", cmd)
        self.assertIn("--checkpoint", cmd)
        self.assertIn(checkpoint_path, cmd)

    @patch("tcrmd.sequence_to_structure.subprocess.run")
    def test_extra_args_forwarded(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        input_path, checkpoint_path = self._create_dummy_files()

        RunBoltzInference(
            input_path,
            checkpoint_path,
            self.tmp_dir,
            extraArgs=["--device", "cpu"],
        )
        cmd = mock_run.call_args[0][0]
        self.assertIn("--device", cmd)
        self.assertIn("cpu", cmd)

    @patch("tcrmd.sequence_to_structure.subprocess.run")
    def test_returns_predicted_pdb_path(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        input_path, checkpoint_path = self._create_dummy_files()
        out_dir = os.path.join(self.tmp_dir, "out")

        result = RunBoltzInference(input_path, checkpoint_path, out_dir)
        self.assertEqual(os.path.basename(result), "predicted_complex.pdb")


class TestAlignToTemplate(unittest.TestCase):
    """Tests for AlignToTemplate."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def _write_dummy_pdb(self, filename: str, offset: float = 0.0) -> str:
        """Write a minimal PDB with 5 CA atoms at known positions."""
        path = os.path.join(self.tmp_dir, filename)
        lines = ["REMARK dummy pdb\n"]
        for i in range(1, 6):
            x = float(i) + offset
            y = float(i) + offset
            z = float(i) + offset
            lines.append(
                f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        lines.append("END\n")
        with open(path, "w") as fh:
            fh.writelines(lines)
        return path

    def test_identical_structures_rmsd_zero(self):
        pdb_a = self._write_dummy_pdb("pred.pdb", offset=0.0)
        pdb_b = self._write_dummy_pdb("tmpl.pdb", offset=0.0)
        out_pdb = os.path.join(self.tmp_dir, "aligned.pdb")
        rmsd = AlignToTemplate(pdb_a, pdb_b, out_pdb)
        self.assertAlmostEqual(rmsd, 0.0, places=4)

    def test_translated_structure_rmsd_small_after_alignment(self):
        pdb_a = self._write_dummy_pdb("pred2.pdb", offset=0.0)
        pdb_b = self._write_dummy_pdb("tmpl2.pdb", offset=5.0)
        out_pdb = os.path.join(self.tmp_dir, "aligned2.pdb")
        rmsd = AlignToTemplate(pdb_a, pdb_b, out_pdb)
        self.assertAlmostEqual(rmsd, 0.0, places=4)

    def test_output_file_created(self):
        pdb_a = self._write_dummy_pdb("pred3.pdb")
        pdb_b = self._write_dummy_pdb("tmpl3.pdb")
        out_pdb = os.path.join(self.tmp_dir, "aligned3.pdb")
        AlignToTemplate(pdb_a, pdb_b, out_pdb)
        self.assertTrue(os.path.isfile(out_pdb))

    def test_missing_predicted_pdb_raises(self):
        pdb_b = self._write_dummy_pdb("tmpl4.pdb")
        with self.assertRaises(FileNotFoundError):
            AlignToTemplate(
                "/nonexistent/pred.pdb",
                pdb_b,
                os.path.join(self.tmp_dir, "out.pdb"),
            )

    def test_missing_template_pdb_raises(self):
        pdb_a = self._write_dummy_pdb("pred5.pdb")
        with self.assertRaises(FileNotFoundError):
            AlignToTemplate(
                pdb_a,
                "/nonexistent/tmpl.pdb",
                os.path.join(self.tmp_dir, "out.pdb"),
            )

    def test_no_common_residues_raises(self):
        """PDBs whose residue numbers don't overlap should raise RuntimeError."""
        path_a = os.path.join(self.tmp_dir, "pdb_no_common_a.pdb")
        path_b = os.path.join(self.tmp_dir, "pdb_no_common_b.pdb")
        with open(path_a, "w") as fh:
            fh.write(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
            )
        with open(path_b, "w") as fh:
            fh.write(
                "ATOM      1  CA  ALA A 999       0.000   0.000   0.000\n"
            )
        with self.assertRaises(RuntimeError):
            AlignToTemplate(
                path_a, path_b, os.path.join(self.tmp_dir, "out_nc.pdb")
            )

    def test_rmsd_is_float(self):
        pdb_a = self._write_dummy_pdb("pred6.pdb")
        pdb_b = self._write_dummy_pdb("tmpl6.pdb")
        out_pdb = os.path.join(self.tmp_dir, "aligned6.pdb")
        rmsd = AlignToTemplate(pdb_a, pdb_b, out_pdb)
        self.assertIsInstance(rmsd, float)

    def test_output_contains_atom_records(self):
        pdb_a = self._write_dummy_pdb("pred7.pdb")
        pdb_b = self._write_dummy_pdb("tmpl7.pdb")
        out_pdb = os.path.join(self.tmp_dir, "aligned7.pdb")
        AlignToTemplate(pdb_a, pdb_b, out_pdb)
        with open(out_pdb) as fh:
            content = fh.read()
        self.assertIn("ATOM", content)


class TestCDRRegionsDefinition(unittest.TestCase):
    """Validate the _CDR_REGIONS constant itself."""

    def test_both_chains_present(self):
        self.assertIn("alpha", _CDR_REGIONS)
        self.assertIn("beta", _CDR_REGIONS)

    def test_three_regions_per_chain(self):
        for chain in ("alpha", "beta"):
            self.assertEqual(len(_CDR_REGIONS[chain]), 3)

    def test_cdr3_starts_after_cdr2(self):
        for chain in ("alpha", "beta"):
            cdr2_end = _CDR_REGIONS[chain]["CDR2"][1]
            cdr3_start = _CDR_REGIONS[chain]["CDR3"][0]
            self.assertGreater(cdr3_start, cdr2_end)

    def test_region_start_less_than_end(self):
        for chain in ("alpha", "beta"):
            for region, (start, end) in _CDR_REGIONS[chain].items():
                self.assertLess(start, end, f"{chain}/{region}: start must be < end")


if __name__ == "__main__":
    unittest.main()
