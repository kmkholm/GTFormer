import os, time, subprocess, sys
CWD=os.environ.get("GTF_ROOT", ".")
def sh(args):
    print(">>", " ".join(args), flush=True)
    subprocess.run(args, cwd=CWD)
# wait for vectorization of all formats
while "VEC ALL DONE" not in open(os.path.join(CWD,"vec.log")).read():
    time.sleep(60)
sh(["python","prep.py"])
runs=[("gtformer","full"),("mlp","mlp_plain"),("moe","full"),("proto","full"),
      ("gtformer","plain"),("gtformer","no_cam"),("gtformer","no_grl"),("gtformer","no_aux"),
      ("mlp","full"),("moe","plain"),("proto","plain"),("proto","no_con")]
for arch,v in runs:
    tag=f"{arch}_{v}"
    if os.path.exists(os.path.join(CWD,"results",f"scores_Win32_{tag}.npz")): continue
    sh(["python","gtformer.py",v,"--arch",arch,"--tag",tag])
# extra seeds for the three proposed architectures (mean +- sd)
for seed in (1,):
    for arch in ("gtformer","moe","proto"):
        tag=f"{arch}_full_s{seed}"
        if os.path.exists(os.path.join(CWD,"results",f"scores_Win32_{tag}.npz")): continue
        sh(["python","gtformer.py","full","--arch",arch,"--tag",tag,"--seed",str(seed)])
print("DEEP ALL DONE", flush=True)
