from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

import numpy as np
import torch

from nnunetv2.inference.microbt_pf_predictor import (
    MicroBTPFPredictor,
    raw_logits_to_microbt_pf,
)
from nnunetv2.topology.persistent_field_0d import decode_persistent_field
from nnunetv2.training.dataloading.microbt_pf_loader import MicroBTPFDataLoader
from nnunetv2.training.dataloading.microbt_pf_targets import label_instances_26
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTPFTrainerA0,
    MicroBTPFTrainerA3,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


DATASET_NAME = "Dataset001_Example"
PROTECTED_SOURCE = Path("/path/to/example")
PROTECTED_RESULT_ROOTS = (
    Path("/path/to/example"),
    Path("/path/to/example"),
    Path("/path/to/example"),
    Path("/path/to/example"),
    Path("/path/to/example"),
)


def select_cached_cases(cache_directory: Path, count: int) -> tuple[str, ...]:
    cases = tuple(sorted(path.name for path in Path(cache_directory).iterdir() if path.is_dir()))
    if count <= 0 or len(cases) < count:
        raise ValueError(f"requested {count} cases but cache contains {len(cases)}")
    return cases[:count]


def feasibility_gate(a0: dict, a3: dict) -> dict:
    memory_ratio = float(a3["max_memory_allocated_bytes"]) / max(
        1, float(a0["max_memory_allocated_bytes"])
    )
    time_ratio = float(a3["median_step_time_seconds"]) / max(
        1e-12, float(a0["median_step_time_seconds"])
    )
    reasons: list[str] = []
    if int(a3["nonfinite_steps"]) != 0:
        reasons.append("a3_nonfinite_steps")
    if int(a3["skipped_updates"]) != 0:
        reasons.append("a3_skipped_updates")
    if memory_ratio > 1.10 + 1e-12:
        reasons.append("memory_ratio_exceeds_1.10")
    if time_ratio > 1.50 + 1e-12:
        reasons.append("step_time_ratio_exceeds_1.50")
    return {
        "memory_ratio_a3_over_a0": memory_ratio,
        "step_time_ratio_a3_over_a0": time_ratio,
        "passed": not reasons,
        "reasons": reasons,
    }


def write_gpu_artifact(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROTECTED_SOURCE), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _protected_snapshot() -> dict:
    status = _git_text("status", "--porcelain=v1")
    process_text = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,lstart=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    protected_processes = [
        line.strip()
        for line in process_text.splitlines()
        if any(str(root) in line for root in (PROTECTED_SOURCE, *PROTECTED_RESULT_ROOTS))
        and "microbt_pf_gpu_feasibility" not in line
    ]
    checkpoint_state: dict[str, dict[str, int]] = {}
    for root in PROTECTED_RESULT_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("checkpoint*.pth"):
            stat = path.stat()
            checkpoint_state[str(path)] = {
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
            }
    return {
        "captured_at": time.time(),
        "source_head": _git_text("rev-parse", "HEAD").strip(),
        "source_dirty_count": len([line for line in status.splitlines() if line]),
        "source_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "protected_processes": protected_processes,
        "protected_checkpoints": checkpoint_state,
    }


def _load_contract(dataset: int, plans_name: str, configuration_name: str):
    if dataset != 1 or plans_name != "nnUNetResEncUNetMPlans" or configuration_name != "3d_fullres":
        raise ValueError("GPU feasibility is frozen to Dataset001 ResEnc-M 3d_fullres")
    dataset_directory = Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME
    plans = json.loads(
        (dataset_directory / f"{plans_name}.json").read_text(encoding="utf-8")
    )
    dataset_json = json.loads(
        (dataset_directory / "dataset.json").read_text(encoding="utf-8")
    )
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    label_manager = plans_manager.get_label_manager(dataset_json)
    cache = (
        Path(os.environ["MICROBT_PF_CACHE"])
        / DATASET_NAME
        / plans_name
        / configuration_name
    )
    source = dataset_directory / "nnUNetPlans_3d_fullres"
    return (
        plans,
        dataset_json,
        plans_manager,
        configuration_manager,
        label_manager,
        cache,
        source,
    )


def _build_loader(
    source: Path,
    cache: Path,
    cases: tuple[str, ...],
    configuration_manager,
    label_manager,
    variant: str,
) -> MicroBTPFDataLoader:
    dataset = build_nnunet_dataset(source, cases)
    component_uniform = variant == "A3"
    return MicroBTPFDataLoader(
        dataset,
        batch_size=1,
        patch_size=tuple(configuration_manager.patch_size),
        final_patch_size=tuple(configuration_manager.patch_size),
        label_manager=label_manager,
        metadata_cache_directory=cache,
        component_uniform=component_uniform,
        oversample_foreground_percent=1.0 if component_uniform else 0.33,
    )


def build_nnunet_dataset(source: Path, cases: tuple[str, ...]):
    dataset_class = infer_dataset_class(str(source))
    # nnUNetDataset sorts this collection in-place during construction.
    return dataset_class(str(source), list(cases))


def _build_variant(
    variant: str,
    configuration_manager,
    seed: int,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.GradScaler]:
    torch.manual_seed(seed)
    builder = MicroBTPFTrainerA0 if variant == "A0" else MicroBTPFTrainerA3
    network = builder.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        4,
        3,
        enable_deep_supervision=False,
    ).cuda()
    trainer = object.__new__(builder)
    criterion = trainer._build_loss().cuda()
    optimizer = torch.optim.SGD(
        network.parameters(), 1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True
    )
    scaler = torch.GradScaler("cuda")
    return network, criterion, optimizer, scaler


def _run_steps(
    variant: str,
    network: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    loader: MicroBTPFDataLoader,
    steps: int,
    seed: int,
) -> tuple[dict, torch.Tensor]:
    np.random.seed(seed)
    random.seed(seed)
    network.train()
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    losses: list[float] = []
    nonfinite = 0
    skipped = 0
    last_data = None
    for step in range(steps):
        batch = loader.generate_train_batch()
        data = batch["data"].cuda(non_blocking=False)
        target = batch["target"].cuda(non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = network(data)
            values = criterion(output, target, batch["persistent_targets"])
            loss = values["loss"]
        if not bool(torch.isfinite(loss)):
            nonfinite += 1
            continue
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in network.parameters()
        )
        if not gradients_finite:
            nonfinite += 1
        torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if step >= min(10, steps // 5):
            times.append(elapsed)
        losses.append(float(loss.detach().cpu()))
        last_data = batch["data"].clone()
    if last_data is None:
        raise RuntimeError(f"{variant} produced no finite optimizer step")
    return (
        {
            "steps": int(steps),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "median_step_time_seconds": float(statistics.median(times or [0.0])),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "nonfinite_steps": int(nonfinite),
            "skipped_updates": int(skipped),
        },
        last_data,
    )


def _save_checkpoint(
    path: Path,
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "network_weights": network.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "grad_scaler_state": scaler.state_dict(),
            "steps": int(steps),
        },
        path,
    )


def _checkpoint_roundtrip(
    output: Path,
    configuration_manager,
    loader: MicroBTPFDataLoader,
    seed: int,
) -> dict:
    network, criterion, optimizer, scaler = _build_variant(
        "A3", configuration_manager, seed
    )
    metrics, fixed_data = _run_steps(
        "A3", network, criterion, optimizer, scaler, loader, 2, seed
    )
    checkpoint = output / "checkpoint_two_step.pth"
    _save_checkpoint(checkpoint, network, optimizer, scaler, 2)
    network.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = network(fixed_data.cuda())
    restored, _, restored_optimizer, restored_scaler = _build_variant(
        "A3", configuration_manager, seed + 1
    )
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    restored.load_state_dict(state["network_weights"])
    restored_optimizer.load_state_dict(state["optimizer_state"])
    restored_scaler.load_state_dict(state["grad_scaler_state"])
    restored.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        observed = restored(fixed_data.cuda())
    bitwise_equal = all(torch.equal(observed[key], expected[key]) for key in expected)
    del network, criterion, optimizer, scaler, restored, restored_optimizer, restored_scaler
    torch.cuda.empty_cache()
    return {
        **metrics,
        "checkpoint": str(checkpoint),
        "reload_bitwise_equal": bool(bitwise_equal),
    }


def _full_volume_metrics(
    network: torch.nn.Module,
    dataset,
    cases: tuple[str, ...],
    configuration_manager,
    label_manager,
) -> list[dict]:
    predictor = MicroBTPFPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.network = network
    predictor.configuration_manager = configuration_manager
    predictor.label_manager = label_manager
    predictor.allowed_mirroring_axes = None
    metrics: list[dict] = []
    network.eval()
    for identifier in cases:
        data, segmentation, previous, _ = dataset.load_case(identifier)
        if previous is not None:
            raise RuntimeError("five-case feasibility does not support cascaded data")
        raw = predictor.predict_sliding_window_return_logits(
            torch.from_numpy(np.asarray(data[:])).float()
        ).float().cpu()
        converted = raw_logits_to_microbt_pf(raw, a0=False)
        field = converted["field"].numpy()
        predicted_wt = field >= 0.0
        target_wt = np.isin(np.asarray(segmentation[0]), (1, 2, 3))
        intersection = int(np.logical_and(predicted_wt, target_wt).sum())
        denominator = int(predicted_wt.sum() + target_wt.sum())
        dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
        predicted_count = int(decode_persistent_field(field)["count"])
        target_count = int(label_instances_26(target_wt).max())
        metrics.append(
            {
                "case": identifier,
                "wt_dice": float(dice),
                "predicted_count": predicted_count,
                "target_count": target_count,
                "exact_count": predicted_count == target_count,
            }
        )
    return metrics


def _memorization_probe(
    network,
    criterion,
    optimizer,
    scaler,
    loader,
    dataset,
    cases,
    configuration_manager,
    label_manager,
    seed: int,
    starting_steps: int,
) -> dict:
    maximum_steps = int(os.environ.get("MICROBT_PF_MEMORIZATION_MAX_STEPS", "1000"))
    completed_steps = starting_steps
    evaluations: list[dict] = []
    while True:
        case_metrics = _full_volume_metrics(
            network, dataset, cases, configuration_manager, label_manager
        )
        passed = all(
            item["wt_dice"] > 0.95 and item["exact_count"]
            for item in case_metrics
        )
        evaluations.append(
            {"step": int(completed_steps), "cases": case_metrics, "passed": passed}
        )
        if passed or completed_steps >= maximum_steps:
            return {
                "attempted": True,
                "passed": passed,
                "completed_steps": int(completed_steps),
                "maximum_steps": int(maximum_steps),
                "evaluations": evaluations,
            }
        additional = min(100, maximum_steps - completed_steps)
        _run_steps(
            "A3",
            network,
            criterion,
            optimizer,
            scaler,
            loader,
            additional,
            seed + completed_steps,
        )
        completed_steps += additional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded MicroBT-PF GPU feasibility")
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--variants", nargs=2, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case-count", type=int, required=True)
    parser.add_argument("--a0-output", type=Path, required=True)
    parser.add_argument("--a3-output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.variants != ["A0", "A3"]:
        raise ValueError("GPU feasibility variants must be exactly A0 A3")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    (
        _,
        _,
        _,
        configuration_manager,
        label_manager,
        cache,
        source,
    ) = _load_contract(args.dataset, args.plans, args.configuration)
    cases = select_cached_cases(cache, args.case_count)
    snapshot_before = _protected_snapshot()
    payload = {
        "schema_version": 1,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "seed": int(args.seed),
        "cases": list(cases),
        "device": torch.cuda.get_device_name(0),
        "bf16_supported": True,
        "protected_state_before": snapshot_before,
        "status": "running",
    }
    write_gpu_artifact(args.artifact, payload)

    loader_two = _build_loader(
        source, cache, cases, configuration_manager, label_manager, "A3"
    )
    payload["two_step_checkpoint"] = _checkpoint_roundtrip(
        args.a3_output, configuration_manager, loader_two, args.seed
    )
    write_gpu_artifact(args.artifact, payload)

    a0_loader = _build_loader(
        source, cache, cases, configuration_manager, label_manager, "A0"
    )
    a0_network, a0_loss, a0_optimizer, a0_scaler = _build_variant(
        "A0", configuration_manager, args.seed
    )
    a0_metrics, _ = _run_steps(
        "A0", a0_network, a0_loss, a0_optimizer, a0_scaler, a0_loader, 100, args.seed
    )
    _save_checkpoint(
        args.a0_output / "checkpoint_100.pth",
        a0_network,
        a0_optimizer,
        a0_scaler,
        100,
    )
    payload["a0_100"] = a0_metrics
    del a0_network, a0_loss, a0_optimizer, a0_scaler
    torch.cuda.empty_cache()
    write_gpu_artifact(args.artifact, payload)

    a3_loader = _build_loader(
        source, cache, cases, configuration_manager, label_manager, "A3"
    )
    a3_network, a3_loss, a3_optimizer, a3_scaler = _build_variant(
        "A3", configuration_manager, args.seed
    )
    a3_metrics, _ = _run_steps(
        "A3", a3_network, a3_loss, a3_optimizer, a3_scaler, a3_loader, 100, args.seed
    )
    _save_checkpoint(
        args.a3_output / "checkpoint_100.pth",
        a3_network,
        a3_optimizer,
        a3_scaler,
        100,
    )
    payload["a3_100"] = a3_metrics
    gate = feasibility_gate(a0_metrics, a3_metrics)
    payload["feasibility_gate"] = gate
    write_gpu_artifact(args.artifact, payload)

    if gate["passed"]:
        full_dataset = build_nnunet_dataset(source, cases)
        payload["memorization"] = _memorization_probe(
            a3_network,
            a3_loss,
            a3_optimizer,
            a3_scaler,
            a3_loader,
            full_dataset,
            cases,
            configuration_manager,
            label_manager,
            args.seed,
            100,
        )
    else:
        payload["memorization"] = {
            "attempted": False,
            "passed": False,
            "reason": "feasibility_gate_failed",
        }
    snapshot_after = _protected_snapshot()
    payload["protected_state_after"] = snapshot_after
    payload["protected_state_unchanged"] = (
        snapshot_before["source_head"] == snapshot_after["source_head"]
        and snapshot_before["source_dirty_count"] == snapshot_after["source_dirty_count"]
        and snapshot_before["source_status_sha256"] == snapshot_after["source_status_sha256"]
        and snapshot_before["protected_checkpoints"] == snapshot_after["protected_checkpoints"]
        and snapshot_before["protected_processes"] == snapshot_after["protected_processes"]
    )
    payload["status"] = "complete"
    write_gpu_artifact(args.artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
