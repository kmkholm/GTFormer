"""Run the whole post-processing chain once all score files exist."""
import subprocess, os
CWD = os.environ.get("GTF_ROOT", ".")
for cmd in [["python", "ensemble.py"], ["python", "analyze.py", "scores_*.npz", "metrics_all.csv"], ["python", "stats_tests.py"],
            ["python", "tables.py"], ["python", "figures.py"], ["python", "dataset_stats.py"]]:
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=CWD, capture_output=True, text=True)
    print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if "Warning" not in l and "warn" not in l.lower())[-6000:], flush=True)
print("FINALIZE DONE")
