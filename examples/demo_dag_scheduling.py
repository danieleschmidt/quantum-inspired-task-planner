#!/usr/bin/env python3
"""Demo: 12-task DAG scheduling — QUBO-SA vs CPM.

Task graph (construction project analogy):

    A(3) ──┐
           ├──> E(2) ──┐
    B(2) ──┘           ├──> I(3) ──┐
                       │           ├──> K(2) ──> L(1)
    C(4) ──> F(3) ──> J(2) ──────┘
                 │
    D(2) ──> G(1)┘
           └──> H(2) ──────────────> K (via J)

Actual edges (more precise):
  A → E, B → E
  E → I
  C → F, F → J, F → G
  D → G, D → H
  I → K, J → K
  K → L

Resources: each task requires cpu units; capacity = 4 CPUs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_dag_scheduler import (
    Task, TaskDAG,
    SimulatedAnnealingScheduler,
    CriticalPathBaseline,
)


def build_dag() -> TaskDAG:
    dag = TaskDAG()

    # Tasks: (id, duration, cpu_requirement)
    tasks = [
        Task("A", duration=3, resources={"cpu": 2}),
        Task("B", duration=2, resources={"cpu": 1}),
        Task("C", duration=4, resources={"cpu": 3}),
        Task("D", duration=2, resources={"cpu": 2}),
        Task("E", duration=2, resources={"cpu": 2}),
        Task("F", duration=3, resources={"cpu": 2}),
        Task("G", duration=1, resources={"cpu": 1}),
        Task("H", duration=2, resources={"cpu": 2}),
        Task("I", duration=3, resources={"cpu": 2}),
        Task("J", duration=2, resources={"cpu": 1}),
        Task("K", duration=2, resources={"cpu": 2}),
        Task("L", duration=1, resources={"cpu": 1}),
    ]
    for t in tasks:
        dag.add_task(t)

    # Precedence edges
    edges = [
        ("A", "E"), ("B", "E"),     # E waits for A and B
        ("E", "I"),                  # I waits for E
        ("C", "F"),                  # F waits for C
        ("F", "G"), ("F", "J"),      # G and J wait for F
        ("D", "G"), ("D", "H"),      # G and H wait for D
        ("I", "K"), ("J", "K"),      # K waits for I and J
        ("K", "L"),                  # L waits for K
    ]
    for u, v in edges:
        dag.add_dependency(u, v)

    return dag


def print_schedule(result, label: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Makespan   : {result.makespan} time-steps")
    print(f"  Feasible   : {result.feasible}")
    if result.violations:
        print(f"  Violations :")
        for v in result.violations:
            print(f"    ⚠  {v}")
    print(f"  Schedule   :")
    for tid, start in sorted(result.schedule.items(), key=lambda x: x[1]):
        bar = " " * start + "█" * result.metadata.get("durations", {}).get(tid, 1)
        print(f"    {tid:4s}  t={start:2d}  {bar}")
    if result.solver == "CPM" and "critical_path" in result.metadata:
        cp = " → ".join(result.metadata["critical_path"])
        print(f"  Critical Path: {cp}")
    if result.solver == "SimulatedAnnealing":
        md = result.metadata
        print(f"  QUBO vars  : {md.get('n_vars', '?')}")
        print(f"  QUBO energy: {md.get('qubo_energy', '?'):.2f}")
        print(f"  Elapsed    : {md.get('elapsed_s', '?')}s")


def main() -> None:
    dag = build_dag()
    resource_capacity = {"cpu": 4}

    print(f"\n12-Task DAG  |  {len(dag)} tasks  |  CPU capacity: {resource_capacity['cpu']}")
    print("Dependencies: A→E, B→E, E→I, C→F, F→G, F→J, D→G, D→H, I→K, J→K, K→L")

    # --- CPM baseline (precedence only, no resource awareness) ---
    cpm = CriticalPathBaseline()
    cpm_result = cpm.schedule(dag, resource_capacity)
    # Attach duration info for display
    cpm_result.metadata["durations"] = {t.id: t.duration for t in dag.tasks}
    print_schedule(cpm_result, "CPM Baseline (precedence only, no resource limits)")

    # --- QUBO-SA solver ---
    sa = SimulatedAnnealingScheduler(
        max_timesteps=22,
        T_init=20.0,
        T_final=0.001,
        n_iterations=300_000,
        penalty_start=15.0,
        penalty_prec=20.0,
        penalty_res=10.0,
        alpha=0.1,
        seed=42,
    )
    sa_result = sa.schedule(dag, resource_capacity)
    sa_result.metadata["durations"] = {t.id: t.duration for t in dag.tasks}
    print_schedule(sa_result, "QUBO-SA Solver (precedence + resource constraints)")

    # --- Summary ---
    print(f"\n{'='*55}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  CPM makespan (lower bound) : {cpm_result.makespan}")
    print(f"  QUBO-SA makespan           : {sa_result.makespan}")
    gap = sa_result.makespan - cpm_result.makespan
    pct = 100 * gap / max(cpm_result.makespan, 1)
    print(f"  Gap                        : +{gap} ({pct:.1f}%)")
    print(f"  QUBO-SA feasible           : {sa_result.feasible}")
    print()

    # Resource utilisation per timestep
    print("  CPU utilisation per timestep (QUBO-SA schedule):")
    for t in range(sa_result.makespan):
        used = sum(
            task.resources.get("cpu", 0)
            for task in dag.tasks
            if sa_result.schedule[task.id] <= t < sa_result.schedule[task.id] + task.duration
        )
        bar = "▓" * used + "░" * max(0, resource_capacity["cpu"] - used)
        print(f"    t={t:2d}  [{bar}] {used}/{resource_capacity['cpu']}")

    return 0 if sa_result.feasible else 1


if __name__ == "__main__":
    sys.exit(main())
