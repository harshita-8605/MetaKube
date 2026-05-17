"""Phase 4 full evaluation — gemma4:e2b via ollama as KubeLLM backend."""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from metakube import MetaKube, load_config, build_kubellm
from kubegraph import KubeGraph
from data.synthetic_kfrd import load_kfrd_as_episodes
from eval.aiopslab_adapter import load_aiopslab_queries
from eval.metrics import heuristic_score, aggregate_scores, DiagnosticScore


def run_eval(n_episodes: int = 100, n_queries: int = 30, save: bool = True):
    print("\n" + "="*60)
    print("MetaKube Phase 4 — Full Eval (gemma4:e2b via ollama)")
    print("="*60)
    t_start = time.time()

    cfg = load_config("config/config.yaml")

    # ── Build components ──────────────────────────────────────────
    print("\n[1/5] Building KubeGraph (toy)...")
    kubegraph = KubeGraph.from_toy()
    print(f"      {kubegraph.stats()}")

    print("[2/5] Loading OllamaKubeLLM (gemma4:e2b)...")
    kubellm = build_kubellm(cfg, backend="ollama")

    print("[3/5] Initializing MetaKube...")
    mk = MetaKube(cfg, kubellm, kubegraph)

    # ── Seed EPMN ────────────────────────────────────────────────
    print(f"[4/5] Seeding EPMN with {n_episodes} KFRD episodes...")
    episodes = load_kfrd_as_episodes(n_samples=n_episodes)
    mk.seed_from_kfrd(episodes)
    print(f"      Episodes={len(mk.epmn.episodes)}  Patterns={len(mk.epmn.patterns)}")

    # ── Eval loop ─────────────────────────────────────────────────
    queries = load_aiopslab_queries(max_problems=n_queries)
    print(f"[5/5] Evaluating {len(queries)} AIOpsLab queries...\n")

    scores: list[DiagnosticScore] = []
    results_log = []

    for i, q in enumerate(queries):
        t_q = time.time()
        result = mk.diagnose(q.query, symptoms=q.symptoms)
        score = heuristic_score(result["response"], q.ground_truth_hint, q.fault_category)
        scores.append(score)

        error = 1.0 - (score.avg / 100.0)
        mk.controller.record_outcome(
            pathway=result["pathway"],
            c_max=result["c_max"],
            error=error,
            latency_s=result["latency_s"],
        )

        results_log.append({
            "problem_id": q.problem_id,
            "task_type": q.task_type,
            "fault_category": q.fault_category,
            "pathway": result["pathway"],
            "c_max": result["c_max"],
            "latency_s": result["latency_s"],
            "score": {
                "effectiveness": score.effectiveness,
                "equivalence": score.equivalence,
                "completeness": score.completeness,
                "safety_accuracy": score.safety_accuracy,
                "avg": score.avg,
            },
            "response_preview": result["response"][:300],
        })

        print(f"  [{i+1:02d}/{len(queries)}] {q.problem_id}")
        print(f"         pathway={result['pathway']}  C_max={result['c_max']:.3f}  "
              f"latency={result['latency_s']:.1f}s  avg_score={score.avg:.1f}")
        print(f"         {result['response'][:200].replace(chr(10), ' ')}\n")

    # ── Aggregate ─────────────────────────────────────────────────
    agg = aggregate_scores(scores)
    elapsed = time.time() - t_start

    # Per-category breakdown
    categories = {}
    for r in results_log:
        cat = r["fault_category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["score"]["avg"])

    print("\n" + "="*60)
    print("PHASE 4 RESULTS (heuristic, 0-100 scale)")
    print("="*60)
    print(f"  Effectiveness:   {agg.effectiveness:.1f}")
    print(f"  Equivalence:     {agg.equivalence:.1f}")
    print(f"  Completeness:    {agg.completeness:.1f}")
    print(f"  Safety/Accuracy: {agg.safety_accuracy:.1f}")
    print(f"  Average:         {agg.avg:.1f}")
    print()
    print("Per-category averages:")
    for cat, cat_scores in sorted(categories.items()):
        print(f"  {cat:<18} {sum(cat_scores)/len(cat_scores):.1f}  (n={len(cat_scores)})")
    print()
    ctrl = mk.controller.stats
    print(f"Controller: τ={ctrl['tau']:.3f}  "
          f"intuitive={ctrl['intuitive_count']}  analytical={ctrl['analytical_count']}")
    print(f"EPMN: {len(mk.epmn.episodes)} episodes  {len(mk.epmn.patterns)} patterns")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("="*60)

    if save:
        out_path = Path("/ai-data/datasets/metakube_results")
        out_path.mkdir(parents=True, exist_ok=True)
        report = {
            "backend": "gemma4:e2b via ollama",
            "n_episodes": n_episodes,
            "n_queries": len(queries),
            "aggregate": {
                "effectiveness": agg.effectiveness,
                "equivalence": agg.equivalence,
                "completeness": agg.completeness,
                "safety_accuracy": agg.safety_accuracy,
                "avg": agg.avg,
            },
            "per_category": {cat: sum(s)/len(s) for cat, s in categories.items()},
            "controller": ctrl,
            "elapsed_s": elapsed,
            "results": results_log,
        }
        with open(out_path / "phase4_results.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved → {out_path}/phase4_results.json")

    return agg


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--queries", type=int, default=30)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    run_eval(n_episodes=args.episodes, n_queries=args.queries, save=not args.no_save)
