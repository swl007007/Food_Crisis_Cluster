"""Run/source identity: bundle staging keeps shapefile components readable; reuse refuses config and request drift."""

import json
import tempfile
import unittest
from pathlib import Path

from ipcch_alphaearth import artifacts, preflight, sources
from ipcch_alphaearth.errors import ContractError


def tiny_shapefile(folder: Path) -> Path:
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame({"admin_code": ["101", "102"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326")
    path = folder / "ipcch_admin_geometry_repaired.shp"
    frame.to_file(path, engine="pyogrio")
    return path


def inputs_for(files: dict, bundles: dict | None = None) -> dict:
    bundles = bundles or {}
    return {"inputs_version": "test-v1", "inputs": {name: {"path": str(p), "sha256": artifacts.sha256_file(p), "bundle": bundles.get(name)}
                                                     for name, p in files.items()}}


class BundleStaging(unittest.TestCase):
    def test_staged_shapefile_bundle_is_readable(self):
        import geopandas as gpd

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "src").mkdir()
            shp = tiny_shapefile(tmp / "src")
            files = {f"geometry_{ext}": shp.with_suffix(f".{ext}") for ext in ("shp", "shx", "dbf", "prj", "cpg") if shp.with_suffix(f".{ext}").exists()}
            cfg = inputs_for(files, {name: "geometry" for name in files})
            run = tmp / "run"
            run.mkdir()
            manifest = sources.stage_inputs(run, cfg)
            staged = sources.staged_path(run, manifest, "geometry_shp")
            self.assertEqual(staged.parent.name, "geometry")
            self.assertEqual(staged.name, shp.name)
            frame = gpd.read_file(staged, engine="pyogrio")
            self.assertEqual(frame["admin_code"].tolist(), ["101", "102"])

    def test_colliding_staged_paths_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "a").mkdir(), (tmp / "b").mkdir()
            (tmp / "a" / "x.csv").write_text("1\n"), (tmp / "b" / "x.csv").write_text("2\n")
            cfg = inputs_for({"one": tmp / "a" / "x.csv", "two": tmp / "b" / "x.csv"}, {"one": "g", "two": "g"})
            (tmp / "run").mkdir()
            with self.assertRaises(ContractError):
                sources.stage_inputs(tmp / "run", cfg)


class SourceRequestDrift(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        (t / "a.csv").write_text("x\n1\n")
        (t / "b.csv").write_text("y\n2\n")
        self.files = {"a": t / "a.csv", "b": t / "b.csv"}
        self.run = t / "run"
        self.run.mkdir()
        self.cfg = inputs_for(self.files)
        self.manifest = sources.stage_inputs(self.run, self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_request_reuses(self):
        self.assertEqual(sources.stage_inputs(self.run, self.cfg)["inputs"], self.manifest["inputs"])

    def test_changed_pinned_hash_refused(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg["inputs"]["a"]["sha256"] = "0" * 64
        with self.assertRaises(ContractError):
            sources.stage_inputs(self.run, cfg)

    def test_added_input_refused(self):
        extra = Path(self.tmp.name) / "c.csv"
        extra.write_text("z\n3\n")
        with self.assertRaises(ContractError):
            sources.stage_inputs(self.run, inputs_for({**self.files, "c": extra}))

    def test_changed_version_refused(self):
        cfg = {**self.cfg, "inputs_version": "test-v2"}
        with self.assertRaises(ContractError):
            sources.stage_inputs(self.run, cfg)

    def test_tampered_staged_bytes_refused(self):
        staged = sources.staged_path(self.run, self.manifest, "a")
        staged.write_text("x\n9\n")
        with self.assertRaises(ContractError):
            sources.stage_inputs(self.run, self.cfg)


class ConfigDrift(unittest.TestCase):
    def test_reopen_refuses_config_drift_on_every_entry_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = preflight.init_run(Path(tmp) / "r1")
            artifacts.require_run_dir(run)  # current identity matches
            preflight.init_run(run)
            record = json.loads((run / "run.json").read_text())
            record["config_sha256"]["experiment-contract.json"] = "f" * 64
            (run / "run.json").write_text(json.dumps(record))
            with self.assertRaises(ContractError):
                artifacts.require_run_dir(run)
            with self.assertRaises(ContractError):
                preflight.init_run(run)

    def test_kind_fixed_at_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = preflight.init_run(Path(tmp) / "r2", kind="developer")
            with self.assertRaises(ContractError):
                preflight.init_run(run, kind="formal")

    def test_dropbox_run_refused(self):
        with self.assertRaises(ContractError):
            preflight.init_run(Path("/tmp/Dropbox-like/run"))


if __name__ == "__main__":
    unittest.main()
