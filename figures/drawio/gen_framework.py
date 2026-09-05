"""
Inventory (from manuscript_v1.md / gtformer.py / lamda_exp.py):
 Inputs: EMBER2024 static vectors x ∈ R^2568 (EMBER v3, 6 file formats, 2.626M train / 606K test / 6,315 challenge);
         LAMDA Drebin vectors x ∈ {0,1}^4561 (10 categories; TRAIN 2013-14, IID/NEAR/FAR).
 Preprocessing: SHA-256 de-duplication; sign(x)·log(1+|x|) → standardise (train stats) → clip ±12 → fp16 (EMBER); binary as-is (LAMDA).
 GT-Former: 12 group tokens → per-group MLP (g→192→192) + group embedding, + file-type token + CLS = 14 tokens → 3× pre-norm Transformer
         (4 heads, FFN 384, dropout 0.1) → CLS z ∈ R^192. 2.13M params. Alternatives: FR-MoE (5.28M), ProtoCon-Net (3.65M), MLP (3.58M), LightGBM.
 Objectives: L_det (BCE) [+ CAM m=2, λ_w=1] [+ λ_aux=0.5 consensus head] [+ λ_grl=0.1 family GRL] [+ λ_con=0.3 SupCon]; ablated → plain BCE kept.
 Head: linear 192→1; rank-average with LightGBM (500 rounds, 64 leaves) → threshold at 1% / 0.1% FPR on validation benign.
 Evaluation/XAI: TPR@FPR, challenge@FPR, novel-family / low-consensus strata, drift, bootstrap; CLS attention, SHAP, MoE gates.
"""
from figlib import Fig

f = Fig(page_w=2700, page_h=1400, scale=0.9, name="Framework")   # 3000 x 1545 canvas

# ---------------------------------------------------------------- Stage 1 (green)
f.stage(10, 10, 900, 330, "Stage 1: Static Feature Input and Data Hygiene", "green")
f.box(25, 75, 330, 100, "EMBER2024 v3 vector<br>2,568 dims · 6 file formats<br>2.626M train / 606K test / 6,315 challenge", "gray", fs=16)
f.box(25, 200, 330, 100, "LAMDA Drebin vector<br>4,561 binary dims · 10 categories<br>2013–14 train → 2016–2025 test", "gray", fs=16)
f.box(400, 75, 490, 60, "SHA-256 de-duplication<br>(public release lists every record twice)", "lightgreen", fs=16)
f.box(400, 150, 490, 60, "sign(x)·log(1+|x|) → standardise (train stats)<br>→ clip ±12 → fp16", "lightgreen", fs=16)
f.box(400, 225, 490, 75, "Training-only metadata: AV consensus <i>r</i>,<br>family <i>f</i>, file type <i>t</i>, week / year<br>(never used at inference)", "lightgreen", fs=16)
f.arrow([(355, 125), (378, 125), (378, 105), (400, 105)]); f.arrow([(355, 250), (378, 250), (378, 180), (400, 180)])
f.arrow([(460, 340), (460, 392)])
f.output_label(475, 342, 600, 45, "<i>x</i> ∈ ℝ<sup>2568</sup> (or {0,1}<sup>4561</sup>), <i>t</i>; (<i>r</i>, <i>f</i>) for training only", fs=17)

# ---------------------------------------------------------------- Stage 2 GT-Former (blue)
f.stage(10, 395, 900, 640, "Stage 2: GT-Former (Group-Token Transformer)", "blue")
f.box(25, 460, 260, 80, "Split vector into<br>12 semantic feature groups", "blue", fs=16)
f.stacked_box(25, 575, 260, 80, "Per-group MLP<br>(<i>g</i> → 192 → 192)", "blue", fs=16)
f.box(25, 690, 260, 80, "+ group embedding<br>+ file-type token + CLS<br>(14 tokens × 192)", "lightgreen", fs=16)
f.arrow([(155, 540), (155, 575)]); f.arrow([(155, 655), (155, 690)]); f.arrow([(285, 730), (330, 730)])
f.strip(335, 470, 8)
f.box(330, 575, 300, 80, "Transformer encoder × 3<br>(4 heads, FFN 384, pre-norm, dropout 0.1)", "purple", fs=16)
f.box(330, 690, 300, 80, "CLS pooling<br><i>z</i> ∈ ℝ<sup>192</sup>", "teal", fs=16)
f.arrow([(480, 510), (480, 575)]); f.arrow([(480, 655), (480, 690)])
f.note(660, 460, 230, 120, "2.13M parameters<br>8 epochs · AdamW<br>one-cycle, bf16<br>batch 2,048", fs=16)
f.box(660, 600, 230, 90, "Format-balanced sampling<br><i>p</i> ∝ 1/√<i>n</i><sub>format</sub>", "lightgreen", fs=16)
f.box(25, 800, 865, 215, "<div style='text-align:left'><b>Group tokens (dims):</b> general (7) · byte histogram (256) · byte-entropy (256) · strings (177) · header (74) · sections (224) · imports (1,282) · exports (129) · data directories (34) · rich header (33) · authenticode (8) · pefile warnings (88)<br><br>"
      "<b>LAMDA tokens:</b> ActivityList · URLDomainList · IntentFilterList · ServiceList · RequestedPermissionList · BroadcastReceiverList · RestrictedApiList · SuspiciousApiList · HardwareComponentsList · UsedPermissionsList</div>", "white", fs=15)
f.arrow([(630, 730), (960, 730)])
f.output_label(640, 740, 260, 45, "<i>z</i> ∈ ℝ<sup>192</sup>", fs=17)

# ---------------------------------------------------------------- Stage 3 alternatives (teal)
f.stage(960, 395, 820, 330, "Stage 3: Alternative Encoders (compared)", "teal")
f.box(975, 460, 385, 110, "FR-MoE<br>trunk 2,600→1,024→512 · 6 experts<br>(512→384→192) + shared expert<br>gate on [<i>h</i>; <i>t</i>] · 5.28M params", "teal", fs=16)
f.box(1380, 460, 385, 110, "ProtoCon-Net<br>MLP 2,600→1,024→512→192<br><i>K</i> = 4 prototypes / class, τ = 0.1<br>SupCon head 192→128 · 3.65M", "teal", fs=16)
f.box(975, 590, 385, 100, "MLP baseline<br>2,600→1,024→512→192<br>3.58M params", "lightteal", fs=16)
f.box(1380, 590, 385, 100, "LightGBM (EMBER2024 benchmark)<br>500 rounds · 64 leaves · lr 0.1<br>one model per format", "lightteal", fs=16)
f.arrow([(1370, 725), (1370, 790)])
f.output_label(1385, 735, 380, 45, "logit <i>s</i><sub>m</sub> for every model <i>m</i>", fs=17)

# ---------------------------------------------------------------- Detail panel objectives (lavender)
f.stage(1810, 10, 1175, 715, "Detailed Training Objectives (studied and ablated)", "lavender")
f.dashed_panel(1825, 75, 560, 300, "Consensus-Aware Margin (CAM)")
f.text(1840, 130, 530, 235, "<div style='text-align:left'>evasiveness <i>e</i> = clip(1 − <i>r</i> / <i>r</i><sub>95</sub>, 0, 1)<br><br>"
       "effective logit <i>z</i>′ = <i>z</i> − <i>m</i>·<i>e</i>·<i>y</i>, <i>m</i> = 2<br><br>"
       "weight <i>w</i> = 1 + λ<sub>w</sub>·<i>e</i>·<i>y</i>, λ<sub>w</sub> = 1<br><br>"
       "L<sub>CAM</sub> = Σ <i>w</i>·BCE(<i>z</i>′, <i>y</i>) / Σ <i>w</i></div>", fs=16, align="left")
f.dashed_panel(2410, 75, 560, 300, "Consensus head and family GRL")
f.text(2425, 130, 530, 235, "<div style='text-align:left'>L<sub>aux</sub> = MSE(σ(<i>h</i><sub>aux</sub>(<i>z</i>)), <i>r</i>) on malware, λ<sub>aux</sub> = 0.5<br><br>"
       "family classifier over |V| = 1,329 families<br>through gradient reversal (λ ramps 0 → 1), λ<sub>grl</sub> = 0.1<br><br>"
       "L<sub>con</sub>: supervised contrastive, positives = same family, λ<sub>con</sub> = 0.3</div>", fs=16, align="left")
f.box(1825, 400, 1145, 75, "L = L<sub>det</sub> + λ<sub>aux</sub> L<sub>aux</sub> + λ<sub>grl</sub> L<sub>fam</sub> (+ λ<sub>con</sub> L<sub>con</sub>)<br>ablations: plain BCE · − CAM · − consensus head · − family GRL · − SupCon", "yellow", fs=17)
f.box(1825, 495, 1145, 75, "Finding: every metadata objective lowers TPR at 0.1% FPR (family GRL most)<br>→ plain BCE is kept for the final models", "peach", fs=17)
f.box(1825, 590, 1145, 100, "Rank-ensemble of Optuna-tuned GT-Former and LightGBM: <i>s</i> = ½ [rank(<i>s</i><sub>GT</sub>) + rank(<i>s</i><sub>LGBM</sub>)]<br>Win32 challenge set @ 1% FPR: 63.2% → 68.8% (95% CI +4.7 to +6.6); novel families: 89.1% → 91.8%", "mauve", fs=17)
f.zoom_link((910, 400), (1810, 400)); f.zoom_link((910, 1030), (1810, 725))

# ---------------------------------------------------------------- Stage 4 head (orange)
f.stage(960, 790, 820, 245, "Stage 4: Detection Head and Fusion", "orange")
f.vchain(975, 850, 60, 160, [("Linear (192 → 1)", "lavender"), ("Logit <i>s</i>", "yellow"), ("Rank-average with LightGBM", "green"), ("Threshold @ FPR", "red")], gap=22)
f.text(1330, 850, 440, 160, "<div style='text-align:left'>Threshold chosen on the time-aware<br>validation benign files (last 4 training<br>weeks) for 1% or 0.1% FPR;<br>realised FPR on the future test set<br>is reported</div>", fs=16, align="left")
f.arrow([(1780, 910), (1830, 910)])
f.output_label(1700, 1040, 300, 45, "malicious / benign", fs=17)

# ---------------------------------------------------------------- Stage 5 evaluation & XAI (pink)
f.stage(1830, 790, 1155, 245, "Evaluation and Explainability Module", "pink")
f.box(1845, 850, 360, 165, "<div style='text-align:left'>• TPR @ 1% and 0.1% FPR<br>• challenge-set detection @ FPR<br>• novel-family, no-family,<br>&nbsp;&nbsp;low-consensus strata<br>• weekly / yearly drift, paired bootstrap</div>", "pink", fs=15)
f.box(2225, 850, 360, 165, "<div style='text-align:left'>• CLS attention over group tokens<br>&nbsp;&nbsp;(benign vs malware vs evasive)<br>• SHAP (TreeExplainer) on LightGBM<br>• FR-MoE gate usage per format</div>", "pink", fs=15)
f.box(2605, 850, 365, 165, "<div style='text-align:left'>Three benchmarks:<br>• EMBER2024 (6 formats, 12 test weeks + challenge)<br>• LAMDA (Android, IID / NEAR / FAR)<br>• BODMAS (PE, monthly Apr–Sep 2020)<br>5 seeds · Optuna-tuned GT-Former · bootstrap CIs</div>", "pink", fs=15)

f.save(r"C:\Users\kmkho\Downloads\MalwareResearch2026\figure_drawio\Fig1_Framework.drawio")
print("saved")
