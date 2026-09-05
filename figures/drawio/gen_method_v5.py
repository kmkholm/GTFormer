"""
GT-Former methodology figure, v5: same 3D language as the FedMamba reference (gradient cubes, slanted slabs,
stacked blocks with a multiplier, rounded detail panels) but a completely new palette (navy / teal / wine / gold /
terracotta / olive on warm paper) and corrected wiring:
  x -> Input Layer -> Group Tokenization Layer -> x3 Transformer blocks -> CLS pooling -> head (LayerNorm, Linear d->1)
  -> rank fusion with LightGBM (fed by the raw vector x, bypass line) -> threshold at 1% / 0.1% FPR -> decision.
  Inside a block: LayerNorm -> QKV -> softmax -> (+) residual  ==>  LayerNorm -> Linear d->2d GELU -> Linear -> (+) residual.
Dashed lines are zoom-in links only (slab -> detail panel); every solid arrow is a data path.
"""
from figlib import Fig

f = Fig(page_w=1120, page_h=620, scale=1.0, name="GT-Former")
AR = "fontFamily=Arial;"

# ---------------------------------------------------------------- palette (fill, gradient, stroke)
NAVY = ("#1F3A5F", "#3D6392", "#122238")
TEAL = ("#0F7C7A", "#3FB0AB", "#0A4A49")
WINE = ("#7A1E3A", "#B0405F", "#4A0F22")
GOLD = ("#C99A2E", "#EFCB6C", "#7A5A12")
TERRA = ("#C4552D", "#EA8A64", "#7A3218")
OLIVE = ("#6B7A2A", "#A3B24F", "#3F4A14")
PAPER = ("#FBF8F1", "#F3EEE2", "#3A3A3A")
CREAM = ("#FFF7E6", "#FFFFFF", "#8A6D2E")
MIST = ("#E3EEF0", "#F6FAFB", "#3E6A70")
SAND = ("#F3E2C7", "#FBF1DF", "#8A6D2E")
ROSE = ("#F2D5DC", "#FBEDF1", "#7A1E3A")
GREY = ("#E4E1DA", "#F5F3EE", "#555555")
WHITE = "#FFFFFF"

def V(x, y, w, h, label, style):
    return f.vertex(label, style + AR, x, y, w, h)

def cube(x, y, w, h, pal, label="", size=8, fs=11, sw=1, color="#000000"):
    fill, grad, stroke = pal
    return V(x, y, w, h, label, f"shape=cube;size={size};direction=south;whiteSpace=wrap;html=1;fillColor={fill};gradientColor={grad};strokeColor={stroke};strokeWidth={sw};shadow=1;fontSize={fs};fontColor={color};")

def rbox(x, y, w, h, label, pal, fs=11, arc=10, sw=1, extra="", color="#000000"):
    fill, grad, stroke = pal
    return V(x, y, w, h, label, f"rounded=1;arcSize={arc};whiteSpace=wrap;html=1;fillColor={fill};gradientColor={grad};strokeColor={stroke};strokeWidth={sw};fontSize={fs};fontColor={color};" + extra)

def poly(x, y, w, h, coords, fill, grad, stroke, sw=1.1, shadow=1):
    g = f"gradientColor={grad};" if grad else ""
    return V(x, y, w, h, "", f"shape=mxgraph.basic.polygon;polyCoords={coords};polyline=0;html=1;fillColor={fill};{g}strokeColor={stroke};strokeWidth={sw};shadow={shadow};")

def slab(x, y, w, h, pal, top, side):
    """slanted 3D slab: front face + top face + left side. Front face spans x+15 .. x+15+w."""
    fill, grad, stroke = pal
    poly(x + 15, y, w, h, "[[0.45,0],[1,0.06],[1,0.89],[0,1],[0,0.20]]", fill, grad, stroke, 1.2)
    poly(x + 6, y, w + 9, 44, "[[0,1],[0.52,0],[1,0.30],[0.11,1]]", top, None, stroke)
    poly(x + 6, y + 43, 9, h - 43, "[[0,0],[1,0],[1,1],[0,0.93]]", side, None, stroke)

def T(x, y, w, h, text, fs=11, bold=False, align="center", color="#000000"):
    return f.text(x, y, w, h, text, fs=fs, bold=bold, align=align, color=color, extra=AR)

def A(pts, **kw):
    return f.arrow(pts, **kw)

def circle(cx, cy, r, label, pal, fs=14):
    fill, grad, stroke = pal
    return V(cx - r, cy - r, 2 * r, 2 * r, label, f"ellipse;html=1;fillColor={fill};gradientColor={grad};strokeColor={stroke};strokeWidth=1.2;fontSize={fs};")

def vtext(x, y, w, h, text, fs=11, color=WHITE):
    return T(x, y, w, h, f"<span style='writing-mode:vertical-rl;transform:rotate(180deg)'>{text}</span>", fs=fs, color=color)

# ================================================================== title
T(300, 2, 480, 26, "<b>GT-Former Architecture</b>", fs=18)

# ================================================================== 1. static feature vectors
T(8, 30, 150, 34, "<b>Static feature vectors<br><i>x</i> ∈ ℝ<sup><i>d</i></sup></b>", fs=11)
cube(58, 78, 50, 116, GOLD, size=8, sw=1.1)
cube(71, 83, 50, 116, TEAL, size=8, sw=1.1)
cube(84, 88, 54, 116, NAVY, size=8, sw=1.1)
T(0, 210, 160, 42, "EMBER2024 v3 (2,568)<br>LAMDA Drebin (4,561)<br>BODMAS v2 (2,381)", fs=9, color="#222222")
rbox(40, 262, 34, 24, "PE", NAVY, fs=10, color=WHITE)
rbox(80, 262, 34, 24, "APK", TEAL, fs=10, color=WHITE)
rbox(120, 262, 34, 24, "PDF", GOLD, fs=10, color=WHITE)
A([(140, 146), (162, 146)])                       # x -> Input Layer

# ================================================================== 2. Input Layer slab + Group Tokenization slab
T(135, 38, 80, 30, "<b>Input<br>Layer</b>", fs=11)
slab(150, 75, 60, 200, NAVY, "#6A8DB8", "#0E1B2E")      # front face x 165..225
vtext(168, 120, 55, 130, "<b>Input Layer</b>", fs=10)
A([(226, 175), (248, 175)])                       # Input Layer -> Group Tokenization
T(225, 34, 120, 40, "<b>Group<br>Tokenization Layer</b>", fs=10)
slab(240, 96, 75, 217, TEAL, "#7FD0CB", "#063534")       # front face x 255..330
vtext(258, 150, 70, 130, "<b>Group Tokenization</b>", fs=11)
A([(331, 200), (352, 200)])                       # tokens -> Transformer stack

# ================================================================== 3. x3 stacked Transformer blocks (wine stack)
T(340, 48, 150, 44, "<b>x3 Stacked pre-norm<br>Transformer Blocks</b>", fs=11)
shades = [("#5E1229", "#8C2C4B"), ("#66152D", "#953150"), ("#6E1831", "#9E3655"), ("#741B35", "#A63B5A"), ("#7A1E3A", "#AE4160")]
for k, (fc, gc) in enumerate(shades):
    rbox(352 + 6 * k, 101 + 2 * k, 112, 236, "", (fc, gc, "#2A0A14"), arc=11, sw=1.1, extra="shadow=1;")
rbox(388, 113, 112, 236, "<b>x3</b>", ("#8A2444", "#C04A6C", "#2A0A14"), fs=28, arc=12, sw=1.2, extra="shadow=1;", color=WHITE)
STACK_R, STACK_B = 500, 349

# ================================================================== 4. detail panel: pre-norm Transformer block (top middle)
PX, PY, PW, PH = 520, 40, 285, 270
rbox(PX, PY, PW, PH, "", PAPER, arc=10, sw=1.2)
T(PX, PY + 4, PW, 22, "<b>Pre-norm Transformer Block</b>", fs=13)
A([(STACK_R, 113), (PX, PY + 20)], dashed=True, head=False, sw=0.8)      # zoom links (not data paths)
A([(STACK_R, STACK_B), (PX, PY + PH)], dashed=True, head=False, sw=0.8)
# --- MHSA column
MX, MW = PX + 16, 124
rbox(MX, 70, MW, 194, "", WINE, arc=8, sw=1.2)
T(MX, 72, MW, 18, "<b>Multi-Head Self-Attention</b>", fs=9, color=WHITE)
rbox(MX + 4, 92, MW - 8, 30, "LayerNorm<br><span style='font-size:9px'>pre-norm</span>", SAND, fs=10)
rbox(MX + 4, 128, MW - 8, 40, "Q, K, V projections<br><span style='font-size:9px'>4 heads (8 tuned) · <i>d</i><sub>h</sub> = 48</span>", MIST, fs=10)
rbox(MX + 4, 176, MW - 8, 34, "softmax(<i>QK</i><sup>T</sup>/√<i>d</i><sub>h</sub>) <i>V</i><br><span style='font-size:9px'>attention over 14 tokens</span>", CREAM, fs=10)
MC = MX + MW // 2
circle(MC, 238, 12, "+", GREY, fs=13)
A([(MC, 122), (MC, 128)]); A([(MC, 168), (MC, 176)]); A([(MC, 210), (MC, 226)])
A([(MX + 4, 100), (MX - 12, 100), (MX - 12, 238), (MC - 12, 238)], dashed=True, sw=0.9)   # residual skip around MHSA
# --- FFN column
FX, FW = PX + 152, 124
rbox(FX, 70, FW, 194, "", TEAL, arc=8, sw=1.2)
T(FX, 72, FW, 18, "<b>Feed-Forward</b>", fs=10, color=WHITE)
rbox(FX + 4, 92, FW - 8, 30, "LayerNorm", SAND, fs=10)
rbox(FX + 4, 128, FW - 8, 40, "Linear <i>d</i> → 2<i>d</i> (4<i>d</i> tuned)<br><span style='font-size:9px'>GELU · dropout</span>", MIST, fs=10)
rbox(FX + 4, 176, FW - 8, 34, "Linear 2<i>d</i> → <i>d</i><br><span style='font-size:9px'>back to model width</span>", CREAM, fs=10)
FC = FX + FW // 2
circle(FC, 238, 12, "+", GREY, fs=13)
A([(FC, 122), (FC, 128)]); A([(FC, 168), (FC, 176)]); A([(FC, 210), (FC, 226)])
A([(FX + 4, 100), (FX - 12, 100), (FX - 12, 238), (FC - 12, 238)], dashed=True, sw=0.9)   # residual skip around FFN
# --- MHSA output (+) -> FFN input: routed through the gap between the two columns
GX = (MX + MW + FX) // 2
A([(MC + 12, 238), (GX, 238), (GX, 84), (FX + 4, 84)], sw=1.2)
T(PX + 8, 268, PW - 16, 30, "<span style='font-size:9px'>dashed = residual (identity) paths · solid = data flow · block input enters the MHSA LayerNorm, block output leaves the FFN (+)</span>", fs=9, color="#333333")

# ================================================================== 5. CLS pooling (gold slab) fed from the stack
CX, CY, CW, CH = 425, 372, 84, 150
rbox(CX, CY, CW, CH, "", GOLD, arc=28, sw=1.2, extra="shadow=1;")
vtext(CX + 2, CY + 10, CW - 4, CH - 20, "<b>CLS token pooling</b><br>LayerNorm → <i>z</i> ∈ ℝ<sup><i>d</i></sup>", fs=10, color="#2B1F05")
A([(462, STACK_B), (462, CY)])                    # stack -> CLS pooling
T(CX - 10, CY + CH + 4, CW + 20, 28, "pooled<br>representation <i>z</i>", fs=9)

# ================================================================== 6. Group Tokenization detail panel (bottom-left)
GX0, GY0, GW0, GH0 = 120, 372, 285, 198
rbox(GX0, GY0, GW0, GH0, "", PAPER, arc=12, sw=1.1)
T(GX0, GY0 + 3, GW0, 20, "<b>Group Tokenization Layer</b>", fs=13)
A([(292, 313), (292, GY0)], dashed=True, head=False, sw=0.8)      # zoom link slab -> panel
cube(130, 422, 29, 58, NAVY, size=7)
T(112, 484, 64, 40, "12 semantic<br>groups of <i>x</i>", fs=9)
cube(173, 399, 19, 39, TEAL, size=5)
cube(173, 460, 19, 36, GOLD, size=5)
T(169, 438, 30, 20, "⋮", fs=12)
A([(159, 432), (173, 417)]); A([(159, 462), (173, 477)])
rbox(207, 398, 74, 42, "Linear |<i>g</i><sub>i</sub>| → <i>d</i><br>GELU · Linear <i>d</i> → <i>d</i>", TERRA, fs=9, color=WHITE)
rbox(207, 460, 74, 42, "Linear |<i>g</i><sub>i</sub>| → <i>d</i><br>GELU · Linear <i>d</i> → <i>d</i>", TERRA, fs=9, color=WHITE)
A([(192, 417), (207, 417)]); A([(192, 479), (207, 479)])
cube(298, 397, 30, 45, TEAL, size=6)
cube(298, 460, 30, 43, GOLD, size=6)
A([(281, 419), (298, 419)]); A([(281, 481), (298, 481)])
T(283, 506, 100, 30, "<i>d</i>-dim group tokens<br>(<i>d</i> = 192 / 384)", fs=9)
rbox(340, 402, 58, 96, "+ group<br>embedding<br>+ CLS<br>+ file-type<br>token", MIST, fs=9)
A([(328, 419), (340, 419)]); A([(328, 481), (340, 481)])
T(GX0 + 4, GY0 + GH0 - 24, GW0 - 8, 22, "<span style='font-size:9px'>output: 14 tokens (12 groups + CLS + file type) × <i>d</i> → Transformer stack</span>", fs=9, color="#333333")

# ================================================================== 7. detection head + fusion (bottom middle)
HX, HY, HW, HH = 530, 365, 275, 190
rbox(HX, HY, HW, HH, "", PAPER, arc=12, sw=1.1)
T(HX, HY + 3, HW, 20, "<b>Detection Head &amp; Rank Fusion</b>", fs=13)
A([(CX + CW, 440), (HX, 440)])                    # z -> head
cube(HX + 12, 396, 84, 30, NAVY, "LayerNorm", size=6, fs=10, color=WHITE)
cube(HX + 12, 438, 84, 30, TEAL, "Linear <i>d</i> → 1", size=6, fs=10, color=WHITE)
A([(HX + 54, 426), (HX + 54, 438)])
OX, OY = HX + 160, 453
A([(HX + 96, 453), (OX - 16, 453)])
T(HX + 100, 432, 46, 18, "<i>s</i><sub>GT</sub>", fs=10)
circle(OX, OY, 16, "⊕", GOLD, fs=15)
rbox(OX + 22, OY - 18, 72, 36, "rank average<br>½[r(<i>s</i><sub>GT</sub>)+r(<i>s</i><sub>LGBM</sub>)]", GREY, fs=8)
A([(OX + 16, OY), (OX + 22, OY)])
# LightGBM partner (bottom-left of the panel), fed by the raw vector x; its score enters (+) from below
LX, LY, LW, LH = HX + 12, 496, 110, 36
rbox(LX, LY, LW, LH, "LightGBM<br><span style='font-size:9px'>500 rounds · 64 leaves</span>", OLIVE, fs=10, color=WHITE)
A([(LX + LW, LY + LH // 2), (OX, LY + LH // 2), (OX, OY + 16)])
T(OX + 4, OY + 24, 60, 18, "<i>s</i><sub>LGBM</sub>", fs=10, align="left")
# threshold and decision (right column under the rank-average box)
TX, TY = OX + 22, 494
A([(OX + 58, OY + 18), (OX + 58, TY)])
rbox(TX, TY, 72, 32, "threshold τ<br><span style='font-size:8px'>1% / 0.1% FPR (val)</span>", TERRA, fs=9, color=WHITE)
T(TX - 4, TY + 32, 80, 22, "malicious / benign", fs=9)
# bypass line: raw x -> LightGBM (under both detail panels, enters the head panel from the left)
A([(58, 120), (20, 120), (20, 582), (515, 582), (515, LY + LH // 2), (LX, LY + LH // 2)], sw=0.9, color="#3F4A14")
T(130, 584, 300, 16, "raw feature vector <i>x</i> (no tokenisation) → LightGBM", fs=8, color="#3F4A14")

# ================================================================== 8. right panel: benchmarks, tuning, evaluation
RX, RW = 822, 290
rbox(RX, 2, RW, 600, "", ("#F4F1EA", "#FFFFFF", "none"), arc=8)
T(RX, 8, RW, 40, "<b>Benchmarks, Tuning<br>&amp; Evaluation Protocol</b>", fs=15)
rbox(RX + 10, 58, RW - 20, 226, "", NAVY, arc=10, sw=1.2)
T(RX + 10, 60, RW - 20, 18, "<b>Benchmark Datasets</b>", fs=11, color=WHITE)
for i, (name, rest) in enumerate([("EMBER2024", "3.2M files · 6 formats · 52 wk train / 12 wk test · challenge set 6,315"),
                                   ("LAMDA", "1.0M Android APKs · 2013-2025 · IID / NEAR / FAR"),
                                   ("BODMAS", "134K PE files · Aug 2019-Sep 2020 · monthly test")]):
    rbox(RX + 18, 80 + i * 67, RW - 36, 60, f"<b>{name}</b><br><span style='font-size:9px'>{rest}</span>", CREAM, fs=11, arc=10)
MIDX = RX + RW // 2
A([(MIDX, 284), (MIDX, 296)])
rbox(RX + 14, 296, RW - 28, 42, "<b>Optuna tuning of GT-Former</b><br><span style='font-size:9px'><i>d</i>, depth, heads, FFN, dropout, lr, wd, imports sub-tokens · objective: val ROC-AUC</span>", GOLD, fs=11, arc=10)
A([(MIDX, 338), (MIDX, 350)])
rbox(RX + 14, 350, RW - 28, 40, "", SAND, arc=10)
for k, pal in enumerate([NAVY, TEAL, WINE, GOLD, TERRA]):
    circle(RX + 32 + k * 18, 370, 6, "", pal, fs=8)
T(RX + 112, 351, RW - 126, 38, "<span style='font-size:9px'><b>5 random seeds</b> per deep model<br>paired bootstrap (1,000 resamples)</span>", fs=9)
A([(MIDX, 390), (MIDX, 402)])
rbox(RX + 14, 402, RW - 28, 64, "<b>Operational metrics</b><br><span style='font-size:9px'>TPR @ 1% and 0.1% FPR · challenge-set detection · novel-family, singleton and low-consensus strata · weekly / monthly / yearly drift · confusion matrices, ROC, PR, MCC</span>", MIST, fs=11, arc=12, sw=1.1)
A([(MIDX, 466), (MIDX, 478)])
rbox(RX + 14, 478, RW - 28, 48, "<b>Explainability</b><br><span style='font-size:9px'>CLS attention over group tokens · SHAP on LightGBM · FR-MoE gate usage</span>", ROSE, fs=11, arc=12, sw=1.1)
rbox(RX + 14, 540, RW - 28, 40, "<b>Training</b><br><span style='font-size:9px'>AdamW (lr 6e-4 tuned, one-cycle) · bf16 · single GPU · BCE loss</span>", GREY, fs=11, arc=12, sw=1.1)

f.save(r"C:\Users\kmkho\Downloads\MalwareResearch2026\figure_drawio\Fig2_GTFormer_v5.drawio")
print("saved")
