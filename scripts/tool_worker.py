"""Run native optional tools with their own Python, without changing the app environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
_temporary = Path(__file__).resolve().parents[1] / "output" / "tmp"
_temporary.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(_temporary)
tempfile.tempdir = str(_temporary)


def probe(tool, out):
    result = dict(tool=tool, python=sys.version, executable=sys.executable)
    started = time.perf_counter()
    try:
        if tool == "deeplabcut":
            import deeplabcut
            result["version"] = metadata.version("deeplabcut")
        elif tool == "simba":
            from simba.feature_extractors.feature_extractor_8bp import ExtractFeaturesFrom8bps
            result["version"] = metadata.version("simba-uw-tf-dev")
        elif tool == "labgym":
            from LabGym.categorizer import Categorizers
            import tensorflow as tf
            result.update(version=metadata.version("LabGym"), tensorflow=tf.__version__,
                          gpu=[d.name for d in tf.config.list_physical_devices("GPU")])
        elif tool == "deepethogram":
            import deepethogram
            import torch
            from deepethogram.feature_extractor.models.classifiers import resnet
            result.update(version=metadata.version("deepethogram"), torch=torch.__version__,
                          cuda=torch.cuda.is_available(),
                          gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        result["status"] = "native_import_passed"
    except Exception as e:
        result.update(status="failed", reason=f"{type(e).__name__}: {e}")
    result["seconds"] = time.perf_counter() - started
    path = out / "tools" / f"{tool}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result["status"] != "failed"


def deepethogram_rgb(out):
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from deepethogram.feature_extractor.models.classifiers.resnet import resnet18

    torch.set_num_threads(4)
    torch.hub.set_dir(str((out / "pretrained_weights").resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    records = json.loads((out / "recordings.json").read_text())
    features = out / "rgb_features"
    features.mkdir(exist_ok=True)
    cache = {}
    for rec in records:
        vid = rec["video_id"]
        with np.load(out / "pixel_cache" / f"{vid}.npz") as z:
            rgb, y, t = z["rgb"], z["y"], z["t"]
        path = features / f"{vid}.npz"
        if path.exists():
            with np.load(path) as z:
                x = z["x"]
        else:
            vectors = []
            with torch.inference_mode():
                for start in range(0, len(rgb), 128):
                    batch = torch.from_numpy(rgb[start:start+128].transpose(0, 3, 1, 2).copy()).to(device).float()/255
                    vectors.append(model((batch-mean)/std).cpu().numpy())
            x = np.concatenate(vectors)
            np.savez_compressed(path, x=x)
        cache[vid] = dict(x=x, y=y, t=t)
        print(f"DeepEthogram RGB features {vid}: {x.shape}", flush=True)
    folds = json.loads((out / "splits.json").read_text())["folds"]
    tool = "deepethogram_rgb_probe"
    dest = out / "predictions" / tool
    dest.mkdir(exist_ok=True)
    timings = []
    for fold in folds:
        started = time.perf_counter()
        x = np.concatenate([cache[v]["x"][cache[v]["y"] >= 0] for v in fold["train"]])
        y = np.concatenate([cache[v]["y"][cache[v]["y"] >= 0] for v in fold["train"]])
        head = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced",
                                                                   max_iter=1000, random_state=20261017))
        head.fit(x, y)
        for vid in fold["test"]:
            c = cache[vid]
            p = np.zeros((len(c["y"]), 4), np.float32)
            p[:, head.classes_.astype(int)] = head.predict_proba(c["x"])
            np.savez_compressed(dest / f"{vid}.npz", p=p, y=c["y"], t=c["t"],
                                fold=fold["fold"], trained_on=np.array(fold["train"]))
        timings.append(dict(tool=tool, fold=fold["fold"], training_frames=len(y), feature_count=x.shape[1],
                            test_videos=len(fold["test"]), seconds=time.perf_counter()-started))
        print(f"DeepEthogram RGB probe fold {fold['fold']+1}/5 complete", flush=True)
    import pandas as pd
    (out / "timing").mkdir(exist_ok=True)
    pd.DataFrame(timings).to_csv(out / "timing" / f"{tool}.csv", index=False)
    probe("deepethogram", out)


def labgym_animation(out, epochs=8):
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from LabGym.categorizer import Categorizers

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    records = json.loads((out / "recordings.json").read_text())
    cache = {}
    for rec in records:
        with np.load(out / "pixel_cache" / f"{rec['video_id']}.npz") as z:
            cache[rec["video_id"]] = dict(x=z["animation"], y=z["y"], t=z["t"])
    tool = "labgym_animation_small"
    dest, weights = out / "predictions" / tool, out / "models" / tool
    dest.mkdir(exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    folds = json.loads((out / "splits.json").read_text())["folds"]
    timings = []
    for fold in folds:
        if all((dest / f"{v}.npz").exists() for v in fold["test"]):
            continue
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(20261017)
        started = time.perf_counter()
        x = np.concatenate([cache[v]["x"][cache[v]["y"] >= 0] for v in fold["train"]])
        y = np.concatenate([cache[v]["y"][cache[v]["y"] >= 0] for v in fold["train"]])
        x = x[..., None].astype(np.float32)/255
        inputs = tf.keras.Input(shape=(15, 16, 16, 1))
        model = Categorizers().simple_tvgg(inputs, filters=8, classes=4, level=1, with_classifier=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        class_weights = {c: float(len(y)/(4*max(1, (y == c).sum()))) for c in range(4)}
        history = model.fit(x, y, epochs=epochs, batch_size=128, shuffle=True,
                            class_weight=class_weights, verbose=2)
        model.save(str(weights / f"fold_{fold['fold']}.h5"))
        for vid in fold["test"]:
            c = cache[vid]
            p = model.predict(c["x"][..., None].astype(np.float32)/255, batch_size=128, verbose=0)
            np.savez_compressed(dest / f"{vid}.npz", p=p, y=c["y"], t=c["t"],
                                fold=fold["fold"], trained_on=np.array(fold["train"]))
        timings.append(dict(tool=tool, fold=fold["fold"], training_frames=len(y), feature_count=15*16*16,
                            test_videos=len(fold["test"]), seconds=time.perf_counter()-started))
        (out / "timing").mkdir(exist_ok=True)
        pd.DataFrame(timings).to_csv(out / "timing" / f"{tool}.csv", index=False)
        (weights / f"history_{fold['fold']}.json").write_text(json.dumps(history.history), encoding="utf-8")
        print(f"Native LabGym animation fold {fold['fold']+1}/5 complete", flush=True)
    probe("labgym", out)


def simba_binary(out):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from simba.mixins.train_model_mixin import TrainModelMixin

    mixin = TrainModelMixin()
    records = json.loads((out / "recordings.json").read_text())
    folds = json.loads((out / "splits.json").read_text())["folds"]
    cache = {}
    for rec in records:
        vid = rec["video_id"]
        with np.load(out / "cache" / f"{vid}.npz") as z:
            c = dict(y=z["y"], t=z["t"], frame=z["frame"])
        features = pd.read_csv(out / "simba_project/csv/features_extracted" / f"{vid}.csv", index_col=0)
        columns = [name for name in features if not name.endswith(("_x", "_y", "_p"))]
        c["x"] = features.iloc[c["frame"]][columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
        cache[vid] = c
    tool = "simba_native_binary_rf"
    dest, weights = out / "predictions" / tool, out / "models" / tool
    dest.mkdir(exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "feature_columns.json").write_text(json.dumps(columns, indent=2), encoding="utf-8")
    timings = []
    for fold in folds:
        if all((dest / f"{v}.npz").exists() for v in fold["test"]):
            continue
        started = time.perf_counter()
        chunks, targets = [], []
        for vid in fold["train"]:
            c = cache[vid]
            ix = np.arange(0, len(c["y"]), 5)
            ix = ix[c["y"][ix] >= 0]
            chunks.append(c["x"][ix])
            targets.append(c["y"][ix])
        x, y = np.concatenate(chunks), np.concatenate(targets)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x = imputer.fit_transform(x)
        models = []
        for k in (1, 2, 3):
            model = mixin.clf_define(n_estimators=160, max_depth=18, min_samples_leaf=3,
                                     n_jobs=3, verbose=0, class_weight=None)
            model.set_params(random_state=20261017+k)
            model.fit(x, y == k)
            models.append(model)
        for vid in fold["test"]:
            c = cache[vid]
            test = imputer.transform(c["x"])
            binary = np.column_stack([m.predict_proba(test)[:, 1] for m in models])
            p = np.column_stack([np.prod(1-binary, axis=1), binary]).astype(np.float32)
            np.savez_compressed(dest / f"{vid}.npz", p=p, t=c["t"], y=c["y"], multilabel=True,
                                fold=fold["fold"], trained_on=np.array(fold["train"]))
        joblib.dump(dict(imputer=imputer, models=models, columns=columns),
                    weights / f"fold_{fold['fold']}.joblib")
        timings.append(dict(tool=tool, fold=fold["fold"], training_frames=len(y), feature_count=x.shape[1],
                            test_videos=len(fold["test"]), seconds=time.perf_counter()-started))
        (out / "timing").mkdir(exist_ok=True)
        pd.DataFrame(timings).to_csv(out / "timing" / f"{tool}.csv", index=False)
        print(f"Native SimBA binary RF fold {fold['fold']+1}/5 complete", flush=True)
    probe("simba", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["probe", "simba", "simba-binary", "deepethogram-rgb", "labgym-animation"])
    ap.add_argument("--tool", choices=["simba", "labgym", "deepethogram", "deeplabcut"])
    ap.add_argument("--output", type=Path, default=Path("output/open_field_benchmark"))
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    if a.command == "probe":
        if not a.tool:
            ap.error("probe needs --tool")
        if not probe(a.tool, a.output):
            sys.exit(1)
    elif a.command == "simba":
        from simba.feature_extractors.feature_extractor_8bp import ExtractFeaturesFrom8bps
        started = time.perf_counter()
        ExtractFeaturesFrom8bps(str((a.output / "simba_project/project_config.ini").resolve())).run()
        probe("simba", a.output)
        print(f"Native SimBA feature extraction: {time.perf_counter()-started:.1f}s", flush=True)
    elif a.command == "deepethogram-rgb":
        deepethogram_rgb(a.output)
    elif a.command == "labgym-animation":
        labgym_animation(a.output, a.epochs)
    elif a.command == "simba-binary":
        simba_binary(a.output)


if __name__ == "__main__":
    main()
