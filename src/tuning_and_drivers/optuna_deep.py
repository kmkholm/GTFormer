"""GPU-only Optuna tuning of the deep models (GT-Former, MLP).
ember : GT-Former on a 300K-row training subsample, 3 epochs per trial, objective = validation AUC (weeks 48-51); writes results/optuna_gtformer_ember.json
lamda : GT-Former / MLP in-process, 6 epochs per trial; writes results_lamda/optuna_<arch>.json
bodmas: GT-Former / MLP in-process, 8 epochs per trial; writes results_bodmas/optuna_<arch>.json
Usage: python optuna_deep.py ember|lamda|bodmas <arch> [n_trials]"""
import sys, os, json, re, subprocess, optuna, torch
optuna.logging.set_verbosity(optuna.logging.WARNING)
CWD = os.environ.get("GTF_ROOT", ".")
which, arch = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

def suggest(t, arch):
    hp = {"lr": t.suggest_float("lr", 2e-4, 3e-3, log=True), "wd": t.suggest_float("wd", 1e-6, 1e-2, log=True), "drop": t.suggest_float("drop", 0.0, 0.3)}
    if arch == "gtformer":
        hp.update({"d": t.suggest_categorical("d", [128, 192, 256, 384]), "layers": t.suggest_int("layers", 2, 6), "heads": t.suggest_categorical("heads", [4, 8])})
    return hp

if which == "ember":
    out = os.path.join(CWD, "results", f"optuna_{arch}_ember.json")
    def obj(t):
        hp = suggest(t, arch); args = ["python", "gtformer.py", "plain" if arch == "gtformer" else "mlp_plain", "--arch", arch, "--tag", f"tune_{t.number}", "--sub", "300000", "--epochs", "3", "--no_eval",
                                       "--lr", str(hp["lr"]), "--wd", str(hp["wd"]), "--drop", str(hp["drop"])]
        if arch == "gtformer":
            hp["imports_split"] = t.suggest_categorical("imports_split", [1, 2, 4]); hp["ffn_mult"] = t.suggest_categorical("ffn_mult", [2, 4])
            args += ["--d", str(hp["d"]), "--layers", str(hp["layers"]), "--heads", str(hp["heads"]), "--imports_split", str(hp["imports_split"]), "--ffn_mult", str(hp["ffn_mult"])]
        r = subprocess.run(args, cwd=CWD, capture_output=True, text=True); m = re.search(r"BEST_VAL_AUC ([0-9.]+)", r.stdout)
        if not m: print(r.stdout[-800:], r.stderr[-800:]); return 0.0
        print(f"trial {t.number} {hp} -> {m.group(1)}", flush=True); return float(m.group(1))
else:
    import lamda_exp
    mod = lamda_exp if which == "lamda" else __import__("bodmas_exp")
    if which == "lamda":
        X, M, tr, va, ev = lamda_exp.load(); P = X
    else:
        X, M, tr, va, ev = mod.load(); P = mod.prep(X, tr)
    out = os.path.join(mod.RES, f"optuna_{arch}.json"); ep = 6 if which == "lamda" else 8
    def obj(t):
        hp = suggest(t, arch); hp["tuning"] = True; lamda_exp.HP.clear(); lamda_exp.HP.update(hp)
        if which == "lamda": auc = lamda_exp.run_deep(P, M, tr, va, ev, arch, "plain", f"tune_{t.number}", 0, ep)
        else: auc = mod.run_deep(P, M, tr, va, ev, arch, f"tune_{t.number}", 0, ep)
        torch.cuda.empty_cache(); print(f"trial {t.number} {hp} -> {auc:.5f}", flush=True); return auc

st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)); st.optimize(obj, n_trials=N)
bp = dict(st.best_trial.params); bp["val_auc"] = st.best_value; json.dump(bp, open(out, "w")); print("BEST", json.dumps(bp), flush=True)
