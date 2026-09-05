import subprocess, os, json
CWD=os.environ.get("GTF_ROOT", ".")
def sh(a): print(">>", " ".join(map(str,a)), flush=True); subprocess.run(a, cwd=CWD)
def exists(*p): return os.path.exists(os.path.join(CWD,*p))
# 1) BODMAS reference runs (GPU): benchmark LightGBM (OpenCL) + default deep models
for m in ("lgbm","mlp","gtformer","proto"): sh(["python","bodmas_exp.py",m])
# 2) Optuna on the deep models (GPU)
for ds,arch,n in (("bodmas","gtformer",25),("lamda","gtformer",25),("ember","gtformer",20)):
    f = os.path.join(CWD, "results" if ds=="ember" else f"results_{ds}", f"optuna_{arch}_ember.json" if ds=="ember" else f"optuna_{arch}.json")
    if not os.path.exists(f): sh(["python","optuna_deep.py",ds,arch,str(n)])
# 3) retrain tuned configs with 5 seeds
for ds in ("bodmas","lamda"):
    for arch in ("gtformer",):
        hp = os.path.join(CWD, f"results_{ds}", f"optuna_{arch}.json")
        for seed in range(5):
            tag = f"{arch}_tuned" + (f"_s{seed}" if seed else "")
            if not exists(f"results_{ds}", f"scores_{tag}.npz"):
                sh(["python", f"{ds}_exp.py", arch, "--seed", str(seed), "--hp", hp, "--tag", tag] + (["--epochs","12"] if ds=="lamda" else ["--epochs","15"]))
for arch, seeds in (("gtformer", range(5)),):
    hp = json.load(open(os.path.join(CWD, "results", f"optuna_{arch}_ember.json")))
    for seed in seeds:
        tag = f"{arch}_tuned" + (f"_s{seed}" if seed else "")
        if exists("results", f"scores_Win32_{tag}.npz"): continue
        args = ["python","gtformer.py","plain" if arch=="gtformer" else "mlp_plain","--arch",arch,"--tag",tag,"--seed",str(seed),"--lr",str(hp["lr"]),"--wd",str(hp["wd"]),"--drop",str(hp["drop"])]
        if arch=="gtformer": args += ["--d",str(hp["d"]),"--layers",str(hp["layers"]),"--heads",str(hp["heads"]),"--imports_split",str(hp["imports_split"]),"--ffn_mult",str(hp["ffn_mult"])]
        sh(args)
# 4) extra seeds (3,4) for the existing plain models
sh(["python","seeds_driver.py"])
print("GPU MASTER2 DONE", flush=True)
