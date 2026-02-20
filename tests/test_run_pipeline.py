"""
Tests for run_pipeline.py – the end-to-end orchestration script.

All heavy dependencies (Boltz, OpenMM, MDAnalysis, pdbfixer, propka) are
mocked to allow fast unit-level testing of the orchestration logic.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

import run_pipeline


class TestArgParser(unittest.TestCase):
    """Validate the CLI argument parser."""

    def test_requires_sequences(self):
        with self.assertRaises(SystemExit):
            run_pipeline._parse_args([])

    def test_default_output(self):
        args = run_pipeline._parse_args(["--sequences", "seq.json"])
        self.assertEqual(args.output, "results")

    def test_default_platform(self):
        args = run_pipeline._parse_args(["--sequences", "seq.json"])
        self.assertEqual(args.platform, "CPU")

    def test_default_num_steps(self):
        args = run_pipeline._parse_args(["--sequences", "seq.json"])
        self.assertEqual(args.num_steps, 50_000)

    def test_default_temperature(self):
        args = run_pipeline._parse_args(["--sequences", "seq.json"])
        self.assertAlmostEqual(args.temperature, 300.0)

    def test_default_ph(self):
        args = run_pipeline._parse_args(["--sequences", "seq.json"])
        self.assertAlmostEqual(args.ph, 7.4)

    def test_no_hmr_flag(self):
        args = run_pipeline._parse_args(["--sequences", "seq.json", "--no-hmr"])
        self.assertTrue(args.no_hmr)

    def test_skip_inference_flag(self):
        args = run_pipeline._parse_args(
            ["--sequences", "seq.json", "--skip-inference"]
        )
        self.assertTrue(args.skip_inference)

    def test_custom_output(self):
        args = run_pipeline._parse_args(
            ["--sequences", "seq.json", "--output", "my_results"]
        )
        self.assertEqual(args.output, "my_results")

    def test_custom_template(self):
        args = run_pipeline._parse_args(
            ["--sequences", "seq.json", "--template", "my_template.pdb"]
        )
        self.assertEqual(args.template, "my_template.pdb")


class TestRunPipelineFunction(unittest.TestCase):
    """Tests for the RunPipeline orchestration function."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.seq_file = os.path.join(self.tmp, "sequences.json")
        with open(self.seq_file, "w") as fh:
            json.dump(
                {
                    "alpha": "A" * 150,
                    "beta": "C" * 150,
                    "peptide": "SIINFEKL",
                },
                fh,
            )

    def _build_mock_modules(self, tmp_dir):
        """Return a dict of mocked module references."""
        mock_s2s = MagicMock()
        mock_s2s.ExtractCDRLoops.return_value = {"CDR1": "ACDE", "CDR2": "FGH", "CDR3": "IKLM"}
        mock_s2s.FormatBoltzInput.return_value = os.path.join(tmp_dir, "boltz_input.yaml")
        # Create a fake input file so downstream code doesn't fail on file checks.
        with open(mock_s2s.FormatBoltzInput.return_value, "w") as fh:
            fh.write("{}")
        mock_s2s.RunBoltzInference.return_value = os.path.join(tmp_dir, "predicted.pdb")
        with open(mock_s2s.RunBoltzInference.return_value, "w") as fh:
            fh.write("END\n")
        mock_s2s.AlignToTemplate.return_value = 1.5

        mock_sp = MagicMock()
        cleaned_pdb = os.path.join(tmp_dir, "cleaned.pdb")
        protonated_pdb = os.path.join(tmp_dir, "protonated.pdb")
        solvated_pdb = os.path.join(tmp_dir, "solvated.pdb")
        for p in (cleaned_pdb, protonated_pdb, solvated_pdb):
            with open(p, "w") as fh:
                fh.write("END\n")
        mock_sp.CleanPDB.return_value = cleaned_pdb
        mock_sp.AssignProtonationStates.return_value = protonated_pdb
        mock_sp.SolvateSystem.return_value = solvated_pdb

        sim_dir = os.path.join(tmp_dir, "simulation")
        os.makedirs(sim_dir, exist_ok=True)
        minimized_pdb = os.path.join(sim_dir, "minimized.pdb")
        traj_dcd = os.path.join(sim_dir, "trajectory.dcd")
        for p in (minimized_pdb, traj_dcd):
            with open(p, "w") as fh:
                fh.write("END\n")
        mock_sim = MagicMock()
        mock_sim.RunSimulation.return_value = {
            "minimized_pdb": minimized_pdb,
            "trajectory": traj_dcd,
            "potential_energy": -12345.0,
        }

        mock_ia = MagicMock()
        mock_ia.ComputeRMSF.return_value = ([], [])
        mock_ia.ComputeHydrogenBondMap.return_value = MagicMock()
        mock_ia.ComputeContactMap.return_value = ([], [], MagicMock())
        mock_ia.ComputeBuriedSurfaceArea.return_value = ([], [])
        mock_ia.ComputeCOMDistance.return_value = ([], [])

        return mock_s2s, mock_sp, mock_sim, mock_ia

    def test_creates_output_directory(self):
        mock_s2s, mock_sp, mock_sim, mock_ia = self._build_mock_modules(self.tmp)
        output_dir = os.path.join(self.tmp, "new_output")

        with patch.dict(
            "sys.modules",
            {
                "tcrmd.sequence_to_structure": mock_s2s,
                "tcrmd.system_preparation": mock_sp,
                "tcrmd.simulate": mock_sim,
                "tcrmd.inference_analytics": mock_ia,
            },
        ):
            run_pipeline.RunPipeline(
                self.seq_file,
                output_dir,
                skipInference=True,
                templatePath=os.path.join(self.tmp, "predicted.pdb"),
            )

        self.assertTrue(os.path.isdir(output_dir))

    def test_returns_dict_with_expected_keys(self):
        mock_s2s, mock_sp, mock_sim, mock_ia = self._build_mock_modules(self.tmp)
        output_dir = os.path.join(self.tmp, "results_keys")

        with patch.dict(
            "sys.modules",
            {
                "tcrmd.sequence_to_structure": mock_s2s,
                "tcrmd.system_preparation": mock_sp,
                "tcrmd.simulate": mock_sim,
                "tcrmd.inference_analytics": mock_ia,
            },
        ):
            result = run_pipeline.RunPipeline(
                self.seq_file,
                output_dir,
                skipInference=True,
                templatePath=os.path.join(self.tmp, "predicted.pdb"),
            )

        required_keys = {
            "predicted_pdb",
            "cleaned_pdb",
            "protonated_pdb",
            "solvated_pdb",
            "minimized_pdb",
            "trajectory",
            "rmsf_csv",
            "hbond_npy",
            "contact_npy",
            "bsa_csv",
            "com_csv",
        }
        self.assertTrue(required_keys.issubset(set(result.keys())))

    def test_writes_pipeline_summary_json(self):
        mock_s2s, mock_sp, mock_sim, mock_ia = self._build_mock_modules(self.tmp)
        output_dir = os.path.join(self.tmp, "results_summary")

        with patch.dict(
            "sys.modules",
            {
                "tcrmd.sequence_to_structure": mock_s2s,
                "tcrmd.system_preparation": mock_sp,
                "tcrmd.simulate": mock_sim,
                "tcrmd.inference_analytics": mock_ia,
            },
        ):
            run_pipeline.RunPipeline(
                self.seq_file,
                output_dir,
                skipInference=True,
                templatePath=os.path.join(self.tmp, "predicted.pdb"),
            )

        summary = os.path.join(output_dir, "pipeline_summary.json")
        self.assertTrue(os.path.isfile(summary))
        with open(summary) as fh:
            data = json.load(fh)
        self.assertIsInstance(data, dict)

    def test_skip_inference_uses_template_directly(self):
        mock_s2s, mock_sp, mock_sim, mock_ia = self._build_mock_modules(self.tmp)
        template_pdb = os.path.join(self.tmp, "predicted.pdb")
        output_dir = os.path.join(self.tmp, "results_skip")

        with patch.dict(
            "sys.modules",
            {
                "tcrmd.sequence_to_structure": mock_s2s,
                "tcrmd.system_preparation": mock_sp,
                "tcrmd.simulate": mock_sim,
                "tcrmd.inference_analytics": mock_ia,
            },
        ):
            result = run_pipeline.RunPipeline(
                self.seq_file,
                output_dir,
                skipInference=True,
                templatePath=template_pdb,
            )

        # RunBoltzInference should NOT have been called.
        mock_s2s.RunBoltzInference.assert_not_called()

    def test_skip_inference_without_template_raises(self):
        mock_s2s, mock_sp, mock_sim, mock_ia = self._build_mock_modules(self.tmp)

        with patch.dict(
            "sys.modules",
            {
                "tcrmd.sequence_to_structure": mock_s2s,
                "tcrmd.system_preparation": mock_sp,
                "tcrmd.simulate": mock_sim,
                "tcrmd.inference_analytics": mock_ia,
            },
        ):
            with self.assertRaises(ValueError):
                run_pipeline.RunPipeline(
                    self.seq_file,
                    os.path.join(self.tmp, "skip_no_tmpl"),
                    skipInference=True,
                    templatePath=None,
                )

    def test_extract_cdr_loops_called_for_each_chain(self):
        mock_s2s, mock_sp, mock_sim, mock_ia = self._build_mock_modules(self.tmp)
        template_pdb = os.path.join(self.tmp, "predicted.pdb")
        output_dir = os.path.join(self.tmp, "results_cdr")

        with patch.dict(
            "sys.modules",
            {
                "tcrmd.sequence_to_structure": mock_s2s,
                "tcrmd.system_preparation": mock_sp,
                "tcrmd.simulate": mock_sim,
                "tcrmd.inference_analytics": mock_ia,
            },
        ):
            run_pipeline.RunPipeline(
                self.seq_file,
                output_dir,
                skipInference=True,
                templatePath=template_pdb,
            )

        # Should have been called once per chain in sequenceData (alpha + beta).
        self.assertEqual(mock_s2s.ExtractCDRLoops.call_count, 2)

    def test_propka_import_error_gracefully_handled(self):
        """Pipeline should continue when propka is absent (logs a warning)."""
        mock_s2s, mock_sp, mock_sim, mock_ia = self._build_mock_modules(self.tmp)
        mock_sp.AssignProtonationStates.side_effect = ImportError("propka not found")
        template_pdb = os.path.join(self.tmp, "predicted.pdb")
        output_dir = os.path.join(self.tmp, "results_nopropka")

        with patch.dict(
            "sys.modules",
            {
                "tcrmd.sequence_to_structure": mock_s2s,
                "tcrmd.system_preparation": mock_sp,
                "tcrmd.simulate": mock_sim,
                "tcrmd.inference_analytics": mock_ia,
            },
        ):
            # Should not raise.
            result = run_pipeline.RunPipeline(
                self.seq_file,
                output_dir,
                skipInference=True,
                templatePath=template_pdb,
            )

        self.assertIn("solvated_pdb", result)


class TestRunPipelineSignature(unittest.TestCase):
    """Verify RunPipeline uses camelCase public arguments."""

    def test_run_pipeline_signature(self):
        import inspect

        params = list(inspect.signature(run_pipeline.RunPipeline).parameters.keys())
        self.assertIn("sequencesPath", params)
        self.assertIn("outputDir", params)
        self.assertIn("templatePath", params)
        self.assertIn("checkpointPath", params)
        self.assertIn("numSteps", params)
        self.assertIn("temperature", params)
        self.assertIn("platformName", params)
        self.assertIn("ph", params)
        self.assertIn("hmr", params)
        self.assertIn("skipInference", params)


if __name__ == "__main__":
    unittest.main()
