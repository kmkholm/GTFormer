import os
import subprocess
CWD=os.environ.get("GTF_ROOT", ".")
runs=[["mlp"],["gtformer"],["proto"],["gtformer","--variant","full"],["mlp","--variant","full"],
      ["gtformer","--seed","1"],["gtformer","--seed","2"],["mlp","--seed","1"],["mlp","--seed","2"],["lgbm","--seed","1"],["lgbm","--seed","2"]]
for r in runs:
    print(">>", r, flush=True); subprocess.run(["python","lamda_exp.py"]+r, cwd=CWD)
print("LAMDA ALL DONE", flush=True)
