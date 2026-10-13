"""Exercise log following and isolation without loading an online SDK or key."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPTS = Path(__file__).parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("swanlab_upload", SCRIPTS / "swanlab_upload.py")
upload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(upload)
METRIC_SPEC = importlib.util.spec_from_file_location("swanlab_metrics", SCRIPTS / "swanlab_metrics.py")
metrics = importlib.util.module_from_spec(METRIC_SPEC)
METRIC_SPEC.loader.exec_module(metrics)


def options(directory, **overrides):
    values = dict(swanlab=True, swanlab_project="fixture-project", swanlab_workspace=None,
        swanlab_run_name=None, swanlab_mode="offline", swanlab_key_file=directory / "missing.key",
        swanlab_python=directory / "missing-python", run_dir=directory, follow_pid=None, force=False)
    return argparse.Namespace(**(values | overrides))


def row(step, **values):
    return dict(vector_step=step, transitions=step * 512, **values)


class SwanLabUploadTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.module_patch = mock.patch.dict(sys.modules, {"swanlab_metrics": metrics})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_group_same_transition_retains_train_eval_episode_and_reward(self):
        records = [row(10, train={"actor_loss": 2}, reward_weighted_terms={"energy_reward": -.1}),
                   row(10, episodes={"count": 2, "mean_f1": .5}),
                   row(10, eval={"f1": .8, "recall": .7, "precision": .9}),
                   row(20, train={"actor_loss": 1})]
        grouped = list(upload.grouped_scalars(records, {}))
        self.assertEqual([step for step, _ in grouped], [5120, 10240])
        self.assertEqual(list(grouped[0][1])[:4], [*metrics.PRIMARY, "metrics/04_energy_reward"])
        self.assertEqual(grouped[0][1]["train/actor_loss"], 2)
        self.assertEqual(grouped[0][1]["train_episodes/mean_f1"], .5)
        self.assertEqual(grouped[0][1]["metrics/01_F1"], .8)
        self.assertEqual(grouped[1][1]["train/actor_loss"], 1)

    def test_grouping_waits_for_next_step_and_flushes_final_step(self):
        consumed = []
        def records():
            for step in (10, 10, 20):
                consumed.append(step)
                yield row(step)
        grouped = upload.grouped_scalars(records(), {})
        self.assertEqual(next(grouped)[0], 5120)
        self.assertEqual(consumed, [10, 10, 20])
        self.assertEqual(next(grouped)[0], 10240)
        with self.assertRaises(StopIteration):
            next(grouped)

    def test_backward_transitions_are_reported(self):
        with self.assertRaisesRegex(ValueError, "backwards"):
            list(upload.grouped_scalars([row(20), row(10)], {}))

    def test_invalid_transition_types_do_not_reach_sdk(self):
        for value in (True, -1, 0.5, "512", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(upload.grouped_scalars([{"transitions": value}], {}))

    def test_import_allows_complete_final_record_without_newline(self):
        path = self.directory / "metrics.jsonl"
        path.write_text(json.dumps(row(10)) + "\n" + json.dumps(row(20)))
        self.assertEqual(list(upload.iter_records(path)), [row(10), row(20)])

    def test_partial_concurrent_line_is_retried_at_same_offset(self):
        path = self.directory / "metrics.jsonl"
        first, second = row(10), row(20)
        encoded = json.dumps(second)
        path.write_text(json.dumps(first) + "\n" + encoded[:12])
        def complete_line(_seconds):
            with path.open("a") as stream:
                stream.write(encoded[12:] + "\n")
        with mock.patch.object(upload, "process_alive", side_effect=[True, False]), \
             mock.patch.object(upload.time, "sleep", side_effect=complete_line) as sleep:
            actual = list(upload.iter_records(path, follow_pid=123, poll_seconds=.01))
        self.assertEqual(actual, [first, second])
        self.assertEqual(sleep.call_count, 1)

    def test_invalid_final_partial_record_raises_instead_of_silently_dropping(self):
        path = self.directory / "metrics.jsonl"
        path.write_text('{"transitions":')
        with self.assertRaises(json.JSONDecodeError):
            list(upload.iter_records(path))

    def test_missing_file_after_learner_exit_does_not_wait_forever(self):
        with mock.patch.object(upload, "process_alive", return_value=False), \
             mock.patch.object(upload.time, "sleep") as sleep, self.assertRaises(FileNotFoundError):
            list(upload.iter_records(self.directory / "missing.jsonl", follow_pid=123))
        sleep.assert_not_called()

    def test_config_waits_for_complete_model_resolution(self):
        path = self.directory / "config.json"
        path.write_text('{"num_envs":')
        writes = iter(({"num_envs": 512}, {"num_envs": 512, "observation_size": 1231}))
        def advance(_seconds):
            path.write_text(json.dumps(next(writes)))
        with mock.patch.object(upload, "process_alive", return_value=True), \
             mock.patch.object(upload.time, "sleep", side_effect=advance) as sleep:
            result = upload.read_config(path, follow_pid=123)
        self.assertEqual(result["observation_size"], 1231)
        self.assertEqual(sleep.call_count, 2)

    def test_missing_uploader_does_not_fail_successful_real_learner(self):
        target = self.directory / "learner_completed"
        command = [sys.executable, "-c", "from pathlib import Path; Path('learner_completed').touch()"]
        with redirect_stderr(io.StringIO()):
            result = upload.run_with_upload(command, root=self.directory, env=os.environ.copy(),
                output=self.directory / "run", options=options(self.directory))
        self.assertEqual(result.returncode, 0)
        self.assertTrue(target.is_file())

    def test_missing_uploader_preserves_learner_failure_code(self):
        command = [sys.executable, "-c", "raise SystemExit(7)"]
        with redirect_stderr(io.StringIO()), self.assertRaises(subprocess.CalledProcessError) as error:
            upload.run_with_upload(command, root=self.directory, env=os.environ.copy(),
                output=self.directory / "run", options=options(self.directory))
        self.assertEqual(error.exception.returncode, 7)

    def test_disabled_reporting_uses_original_checking_subprocess_path(self):
        command = ["learner", "--argument"]
        for opts in (options(self.directory, swanlab=False),
                     options(self.directory, swanlab_mode="disabled")):
            with mock.patch.object(upload.subprocess, "run") as run, \
                 mock.patch.object(upload.subprocess, "Popen") as popen:
                upload.run_with_upload(command, root=self.directory, env={},
                    output=self.directory / "run", options=opts)
            run.assert_called_once_with(command, cwd=self.directory, env={}, check=True)
            popen.assert_not_called()

    def test_uploader_nonzero_exit_does_not_replace_learner_success(self):
        learner, reporter = mock.MagicMock(), mock.MagicMock()
        learner.__enter__.return_value = learner
        learner.wait.return_value = 0
        learner.pid = 123
        reporter.returncode = 19
        with mock.patch.object(upload.subprocess, "Popen", side_effect=[learner, reporter]), \
             redirect_stderr(io.StringIO()) as stderr:
            result = upload.run_with_upload(["learner"], root=self.directory, env={},
                output=self.directory / "run", options=options(self.directory))
        self.assertEqual(result.returncode, 0)
        self.assertIn("needs attention", stderr.getvalue())

    def test_hung_uploader_is_stopped_without_losing_learner_result(self):
        learner, reporter = mock.MagicMock(), mock.MagicMock()
        learner.__enter__.return_value = learner
        learner.wait.return_value = 0
        learner.pid = 123
        reporter.wait.side_effect = [subprocess.TimeoutExpired("uploader", 30),
                                    subprocess.TimeoutExpired("uploader", 5), 0]
        reporter.returncode = -9
        with mock.patch.object(upload.subprocess, "Popen", side_effect=[learner, reporter]), \
             redirect_stderr(io.StringIO()):
            result = upload.run_with_upload(["learner"], root=self.directory, env={},
                output=self.directory / "run", options=options(self.directory))
        self.assertEqual(result.returncode, 0)
        reporter.terminate.assert_called_once()
        reporter.kill.assert_called_once()
        self.assertEqual(reporter.wait.call_args_list,
                         [mock.call(timeout=30), mock.call(timeout=5), mock.call()])

    def test_relative_run_directory_follows_learner_cwd(self):
        learner = mock.MagicMock()
        learner.__enter__.return_value = learner
        learner.wait.return_value = 0
        learner.pid = 123
        with mock.patch.object(upload.subprocess, "Popen", side_effect=[learner, FileNotFoundError]) as popen, \
             redirect_stderr(io.StringIO()):
            upload.run_with_upload(["learner"], root=self.directory, env={},
                output=Path("nested/run"), options=options(self.directory))
        command = popen.call_args_list[1].args[0]
        self.assertEqual(Path(command[command.index("--run-dir") + 1]), self.directory / "nested/run")

    def test_sdk_failure_preserves_metrics_and_redacts_fixture_credential(self):
        # Fake SDK only. No real key is read, and there is no network request.
        (self.directory / "config.json").write_text('{"observation_size": 1231}')
        path = self.directory / "metrics.jsonl"
        original = json.dumps(row(10)) + "\n"
        path.write_text(original)
        fixture_key = "TEST_ONLY_NOT_A_REAL_API_KEY"
        settings = mock.MagicMock()
        fake_sdk = SimpleNamespace(Settings=settings, init=mock.Mock(
            side_effect=RuntimeError("authentication failed: " + fixture_key)))
        with mock.patch.dict(sys.modules, {"swanlab": fake_sdk}), \
             mock.patch.dict(os.environ, {"SWANLAB_API_KEY": fixture_key}), \
             redirect_stderr(io.StringIO()) as stderr, redirect_stdout(io.StringIO()):
            state = upload.upload(options(self.directory, swanlab_mode="online"))
        self.assertEqual(state["status"], "failed")
        self.assertEqual(path.read_text(), original)
        self.assertNotIn(fixture_key, stderr.getvalue())
        saved = (self.directory / "swanlab_upload.json").read_text()
        self.assertNotIn(fixture_key, saved)
        self.assertIn("[REDACTED]", saved)

    def test_offline_upload_never_reads_online_url_and_logs_merged_final_step(self):
        (self.directory / "config.json").write_text('{"observation_size": 1231}')
        records = [row(10, train={"actor_loss": 2}),
                   row(10, eval={"f1": .8, "recall": .7, "precision": .9})]
        (self.directory / "metrics.jsonl").write_text("\n".join(map(json.dumps, records)))
        class OfflineRun:
            id = "fixture-offline-id"
            @property
            def url(self):
                raise ValueError("Offline runs have no URL")
        fake_sdk = SimpleNamespace(Settings=mock.MagicMock(), __version__="fixture-version",
            init=mock.Mock(return_value=OfflineRun()), define_metric=mock.Mock(),
            log=mock.Mock(), finish=mock.Mock())
        with mock.patch.dict(sys.modules, {"swanlab": fake_sdk}), \
             mock.patch.dict(os.environ, {"SWANLAB_API_KEY": ""}), \
             redirect_stdout(io.StringIO()):
            state = upload.upload(options(self.directory))
        self.assertEqual(state["status"], "offline_ready")
        self.assertIsNone(state["url"])
        self.assertEqual(state["records"], 1)
        self.assertEqual(fake_sdk.log.call_args.kwargs, {"step": 5120})
        self.assertEqual(fake_sdk.log.call_args.args[0]["metrics/01_F1"], .8)
        self.assertEqual(fake_sdk.log.call_args.args[0]["train/actor_loss"], 2)
        fake_sdk.finish.assert_called_once()
        self.assertEqual([call.args[0] for call in fake_sdk.define_metric.call_args_list],
                         metrics.metric_order({}))

    def test_completed_upload_dedup_checks_workspace_and_content_not_only_size(self):
        (self.directory / "config.json").write_text('{"observation_size": 1231}')
        path = self.directory / "metrics.jsonl"
        path.write_text(json.dumps(row(10, eval={"f1": .8})) + "\n")
        fake_sdk = SimpleNamespace(Settings=mock.MagicMock(), __version__="fixture-version",
            init=mock.Mock(return_value=SimpleNamespace(url="https://example.invalid/fixture", id="fixture")),
            define_metric=mock.Mock(), log=mock.Mock(), finish=mock.Mock())
        opts = options(self.directory, swanlab_mode="online")
        with mock.patch.dict(sys.modules, {"swanlab": fake_sdk}), \
             mock.patch.dict(os.environ, {"SWANLAB_API_KEY": "TEST_ONLY_NOT_A_REAL_API_KEY"}), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(upload.upload(opts)["status"], "completed")
            upload.upload(opts)
            self.assertEqual(fake_sdk.init.call_count, 1)
            opts.swanlab_workspace = "another-workspace"
            upload.upload(opts)
            self.assertEqual(fake_sdk.init.call_count, 2)
            previous_size = path.stat().st_size
            path.write_text(path.read_text().replace("0.8", "0.9"))
            self.assertEqual(path.stat().st_size, previous_size)
            upload.upload(opts)
            self.assertEqual(fake_sdk.init.call_count, 3)

    def test_entrypoint_consumes_reporting_arguments_without_changing_recipe_or_learner(self):
        spec = importlib.util.spec_from_file_location("train_from_source_fixture", SCRIPTS / "train_from_source.py")
        launcher = importlib.util.module_from_spec(spec)
        # The entrypoint edits sys.path for normal script execution.
        original_path = sys.path[:]
        try:
            spec.loader.exec_module(launcher)
        finally:
            sys.path[:] = original_path
        runtime = SimpleNamespace(make_recipe=mock.Mock(return_value={"fixture_recipe": True}),
            build=mock.Mock(return_value=(self.directory / "scene", self.directory / "task",
                                          self.directory / "source.json")))
        arguments = ["--swanlab", "--swanlab-project", "fixture-project", "--swanlab-workspace", "fixture-team",
            "--swanlab-run-name", "fixture-run", "--swanlab-mode", "offline",
            "--swanlab-key-file", str(self.directory / "fixture.key"),
            "--swanlab-python", str(self.directory / "reporting-python"),
            "--output-dir", str(self.directory / "run"), "--height", ".11", "--num-envs", "512",
            "--seed", "42", "--no-progress"]
        with mock.patch.dict(sys.modules, {"swanlab_upload": upload,
                "tsensei_v3_2_pianist.mjlab_env.source_runtime": runtime}), \
             mock.patch.object(upload, "run_with_upload") as runner, redirect_stderr(io.StringIO()):
            launcher.main(arguments)
        recipe = runtime.make_recipe.call_args.kwargs
        self.assertFalse(any(key == "swanlab" or key.startswith("swanlab_") for key in recipe))
        self.assertEqual(recipe["height"], .11)
        command = runner.call_args.args[0]
        self.assertFalse(any("swanlab" in str(argument) for argument in command))
        self.assertEqual(command[command.index("--num-envs") + 1], "512")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        opts = runner.call_args.kwargs["options"]
        self.assertTrue(opts.swanlab)
        self.assertEqual(opts.swanlab_workspace, "fixture-team")
        self.assertEqual(opts.swanlab_run_name, "fixture-run")
        self.assertEqual(opts.swanlab_mode, "offline")
        runtime.build.assert_called_once_with({"fixture_recipe": True})


if __name__ == "__main__":
    unittest.main()
