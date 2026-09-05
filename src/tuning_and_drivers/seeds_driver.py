import subprocess, os
CWD=os.environ.get("GTF_ROOT", ".")
def sh(a): print(">>", a, flush=True); subprocess.run(a, cwd=CWD)
for seed in (3,4):
    for arch,v in (("gtformer","plain"),("mlp","mlp_plain"),("moe","plain")):
        tag=f"{arch}_{v}_s{seed}"
        if not os.path.exists(os.path.join(CWD,"results",f"scores_Win32_{tag}.npz")): sh(["python","gtformer.py",v,"--arch",arch,"--tag",tag,"--seed",str(seed)])
for seed in (3,4):
    for m in ("gtformer","mlp"):
        sh(["python","lamda_exp.py",m,"--seed",str(seed)])
print("SEEDS ALL DONE", flush=True)
