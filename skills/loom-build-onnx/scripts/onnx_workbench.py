#!/usr/bin/env python3
"""Independent reference/check helpers for single-input, compact FP32 Loom packages."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(command, timeout):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=timeout)
    require(result.returncode == 0,
            f"command failed ({result.returncode}): {command}\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def fetch(args):
    lock = read_json(args.lock)
    out = args.out.resolve()
    if out.exists():
        require(out.stat().st_size == lock["bytes"] and sha256(out) == lock["sha256"],
                f"existing model differs from lock: {out}")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".onnx-download-", dir=out.parent) as temp:
            partial = Path(temp) / "model.onnx"
            run(["curl", "--http1.1", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                 "--connect-timeout", "20", "--max-time", str(args.timeout),
                 lock["url"], "--output", partial], args.timeout * 3 + 90)
            require(partial.stat().st_size == lock["bytes"], "download size differs from lock")
            require(sha256(partial) == lock["sha256"], "download SHA-256 differs from lock")
            require(not out.exists(), f"output appeared during download: {out}")
            partial.rename(out)
    print(json.dumps({"model": str(out), "sha256": lock["sha256"], "bytes": lock["bytes"]}))


def has_external_data(message, onnx):
    if isinstance(message, onnx.TensorProto):
        return message.data_location == onnx.TensorProto.EXTERNAL
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            children = value if field.is_repeated else [value]
            if any(has_external_data(child, onnx) for child in children):
                return True
    return False


def shape_for(meta, override):
    shape = override if override is not None else meta.shape
    require(len(shape) <= 8 and all(type(d) is int and d > 0 for d in shape),
            f"provide a positive static --shape for {meta.name}: {meta.shape}")
    require(len(shape) == len(meta.shape), "input rank differs from ONNX")
    for chosen, original in zip(shape, meta.shape):
        require(not isinstance(original, int) or chosen == original,
                f"--shape overrides fixed ONNX dimension {original} with {chosen}")
    return list(shape)


def write_tensor(root, relative, array, np):
    array = np.array(array, dtype=np.float32, order="C", copy=True)
    require(bool(np.isfinite(array).all()), f"nonfinite reference tensor: {relative}")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return {"file": relative, "dtype": "f32", "shape": list(array.shape), "sha256": sha256(path)}


def reference(args):
    import numpy as np
    import onnx
    import onnxruntime as ort

    model_path = args.model.resolve(strict=True)
    original_hash = sha256(model_path)
    model = onnx.load(model_path, load_external_data=False)
    require(not has_external_data(model, onnx),
            "helper requires a standalone ONNX file; external tensor data needs a separate source lock")
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    require(len(inputs) == 1 and inputs[0].type == "tensor(float)", "helper requires one FP32 input")
    require(outputs and all(x.type == "tensor(float)" for x in outputs), "helper requires FP32 outputs")
    shape = shape_for(inputs[0], args.shape)
    rng = np.random.default_rng(args.seed)
    tensors = [
        ("zeros", np.zeros(shape, dtype=np.float32)),
        ("ones", np.ones(shape, dtype=np.float32)),
        ("normal", rng.standard_normal(shape).astype(np.float32)),
        ("ramp", np.linspace(-1, 1, num=math.prod(shape), dtype=np.float32).reshape(shape)),
    ]
    for index, path in enumerate(args.input):
        require(path.stat().st_size == math.prod(shape) * 4, f"wrong input byte size: {path}")
        tensors.append((f"extra-{index:03d}", np.fromfile(path, dtype=np.float32).reshape(shape)))
    out = args.out.resolve()
    require(not out.exists(), f"reference output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".onnx-reference-", dir=out.parent) as temp:
        root = Path(temp) / "reference"
        root.mkdir()
        cases = []
        output_shapes = None
        for name, tensor in tensors:
            require(bool(np.isfinite(tensor).all()), f"nonfinite input: {name}")
            values = session.run([x.name for x in outputs], {inputs[0].name: tensor})
            current_shapes = [list(value.shape) for value in values]
            require(output_shapes is None or current_shapes == output_shapes, "data-dependent output shapes unsupported")
            output_shapes = current_shapes
            cases.append({
                "name": name,
                "inputs": [write_tensor(root, f"cases/{name}/input-0.bin", tensor, np)],
                "outputs": [write_tensor(root, f"cases/{name}/output-{i}.bin", value, np)
                            for i, value in enumerate(values)],
            })
        require(sha256(model_path) == original_hash, "model changed during reference generation")
        info = {
            "model_sha256": original_hash,
            "ir_version": model.ir_version,
            "opsets": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
            "operator_counts": dict(sorted(Counter(f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in model.graph.node).items())),
            "inputs": [{"name": x.name, "type": x.type, "shape": x.shape} for x in inputs],
            "outputs": [{"name": x.name, "type": x.type, "shape": x.shape} for x in outputs],
            "initializers": [{"name": x.name, "dtype": onnx.TensorProto.DataType.Name(x.data_type), "shape": list(x.dims)}
                             for x in model.graph.initializer],
            "nodes": [{"index": i, "name": n.name, "op": n.op_type, "domain": n.domain,
                       "inputs": list(n.input), "outputs": list(n.output),
                       "attributes": [onnx.helper.printable_attribute(a) for a in n.attribute]}
                      for i, n in enumerate(model.graph.node)],
        }
        manifest = {
            "format": "loom-onnx-reference/1", "model_sha256": original_hash,
            "byteorder": sys.byteorder, "seed": args.seed,
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                            "providers": session.get_providers(), "optimization": "disabled", "threads": 1},
            "inputs": [{"name": inputs[0].name, "dtype": "f32", "shape": shape}],
            "outputs": [{"name": item.name, "dtype": "f32", "shape": dims}
                        for item, dims in zip(outputs, output_shapes)],
            "cases": cases,
        }
        write_json(root / "model-info.json", info)
        write_json(root / "reference.json", manifest)
        require(not out.exists(), f"output appeared during generation: {out}")
        root.rename(out)
    print(json.dumps({"reference": str(out), "model_sha256": original_hash,
                      "inputs": manifest["inputs"], "outputs": manifest["outputs"], "cases": len(cases)}))


def within(root, relative):
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), f"path escapes root: {relative}")
    return path


def read_tensor(root, record, expected, np):
    require(record["dtype"] == expected["dtype"] == "f32" and record["shape"] == expected["shape"],
            "reference tensor shape/dtype differs from profile")
    path = within(root, record["file"])
    require(sha256(path) == record["sha256"], f"reference data SHA-256 differs: {path}")
    require(path.stat().st_size == math.prod(record["shape"]) * 4, f"reference byte size differs: {path}")
    value = np.fromfile(path, dtype=np.float32).reshape(record["shape"])
    require(bool(np.isfinite(value).all()), f"nonfinite reference data: {path}")
    return path, value


def compact_tensor(actual, expected):
    require(actual["dtype"] == expected["dtype"] == "f32" and actual["shape"] == expected["shape"],
            f"package interface differs from ONNX profile: {actual}, expected {expected}")
    compact = []
    stride = 1
    for dim in reversed(expected["shape"]):
        compact.insert(0, stride)
        stride *= dim
    require(not actual.get("strides") or actual["strides"] == compact, "helper requires compact tensors")
    require(actual.get("layout", 0) == 0, "helper requires default layout=0; other layouts need explicit decoding")


def check(args):
    import numpy as np

    require(all(math.isfinite(v) and v >= 0 for v in [args.atol, args.rtol]), "tolerances must be finite and nonnegative")
    report = {"passed": False, "package": str(args.package.resolve()),
              "atol": args.atol, "rtol": args.rtol, "cases": []}
    try:
        reference_root = args.reference.resolve(strict=True)
        package_root = args.package.resolve(strict=True)
        report["model_sha256"] = sha256(args.model)
        report["reference_manifest_sha256"] = sha256(reference_root / "reference.json")
        manifest = read_json(reference_root / "reference.json")
        require(manifest["format"] == "loom-onnx-reference/1", "unknown reference format")
        require(manifest["model_sha256"] == report["model_sha256"], "reference model SHA-256 differs")
        require(manifest["byteorder"] == sys.byteorder, "reference byte order differs from this host")
        require(len(manifest["inputs"]) == 1 and manifest["outputs"] and manifest["cases"], "empty or unsupported reference interface")
        cli = args.cli.resolve(strict=True)
        report["cli"] = str(cli)
        report["validate"] = run([cli, "validate", package_root], args.timeout)
        package = read_json(package_root / "package.json")
        plan = read_json(within(package_root, package["plan"]))
        slots = {x["id"]: x["tensor"] for x in plan["slots"]}
        for direction in ["inputs", "outputs"]:
            require(len(plan[direction]) == len(manifest[direction]), f"{direction} count differs")
            for slot, expected in zip(plan[direction], manifest[direction]):
                compact_tensor(slots[slot], expected)
        for case in manifest["cases"]:
            require(len(case["inputs"]) == 1 and len(case["outputs"]) == len(manifest["outputs"]), "case arity differs")
            input_path, _ = read_tensor(reference_root, case["inputs"][0], manifest["inputs"][0], np)
            expected = [read_tensor(reference_root, x, spec, np)[1]
                        for x, spec in zip(case["outputs"], manifest["outputs"])]
            result = {"name": case["name"], "passed": True, "outputs": []}
            with tempfile.TemporaryDirectory(prefix="loom-output-") as temp:
                result["run"] = run([cli, "run", package_root, temp, input_path], args.timeout)
                for i, target in enumerate(expected):
                    output = Path(temp) / f"{i}.bin"
                    require(output.stat().st_size == target.nbytes, f"output {i} byte size differs")
                    actual = np.fromfile(output, dtype=np.float32).reshape(target.shape)
                    require(bool(np.isfinite(actual).all()), f"nonfinite candidate output {i}")
                    # Compare in float64 so error calculation itself cannot overflow FP32.
                    target64, actual64 = target.astype(np.float64), actual.astype(np.float64)
                    error = np.abs(actual64 - target64)
                    allowed = args.atol + args.rtol * np.abs(target64)
                    bad = int(np.count_nonzero(error > allowed))
                    index = tuple(int(x) for x in np.unravel_index(int(np.argmax(error)), error.shape)) if error.size else ()
                    metrics = {"position": i, "shape": list(target.shape), "failed_elements": bad,
                               "elements": int(error.size), "max_abs_error": float(error.max()) if error.size else 0.0,
                               "max_rel_error": float((error / np.maximum(np.abs(target64), 1e-30)).max()) if error.size else 0.0,
                               "max_abs_index": list(index)}
                    if error.size:
                        metrics.update(reference_at_max_abs=float(target[index]), candidate_at_max_abs=float(actual[index]))
                    if target.ndim == 2 and target.shape[1] > 1:
                        metrics["top1_matches"] = int(np.count_nonzero(np.argmax(actual, axis=1) == np.argmax(target, axis=1)))
                        metrics["top1_total"] = int(target.shape[0])
                    result["outputs"].append(metrics)
                    result["passed"] = result["passed"] and bad == 0
            report["cases"].append(result)
        report["passed"] = all(x["passed"] for x in report["cases"])
    except Exception as error:
        report["error"] = str(error)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(json.dumps({"passed": report["passed"], "report": str(args.report.resolve()),
                      "cases": len(report["cases"]), "error": report.get("error")}))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="download the pinned ResNet-18, verify bytes and SHA-256")
    download.add_argument("--lock", type=Path, default=Path(__file__).resolve().parent.parent / "references/resnet18.lock.json")
    download.add_argument("--out", type=Path, required=True)
    download.add_argument("--timeout", type=int, default=180)
    download.set_defaults(handler=fetch)
    ref = commands.add_parser("reference", help="inspect ONNX and generate independent ORT CPU cases")
    ref.add_argument("--model", type=Path, required=True)
    ref.add_argument("--out", type=Path, required=True)
    ref.add_argument("--shape", type=lambda text: [int(x) for x in text.split(",")])
    ref.add_argument("--seed", type=int, default=0)
    ref.add_argument("--input", type=Path, action="append", default=[], help="additional raw native FP32 input; repeatable")
    ref.set_defaults(handler=reference)
    verify = commands.add_parser("check", help="run Loom package and compare complete outputs to fixed cases")
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--reference", type=Path, required=True)
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--cli", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--atol", type=float, default=1e-4)
    verify.add_argument("--rtol", type=float, default=1e-4)
    verify.add_argument("--timeout", type=int, default=120, help="seconds per CLI command")
    verify.set_defaults(handler=check)
    args = parser.parse_args()
    try:
        return args.handler(args) or 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
