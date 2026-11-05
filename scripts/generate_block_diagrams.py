"""
scripts/generate_block_diagrams.py
Generates two publication-grade, high-resolution block diagrams:
  1. diagram_original_method.png  (Original Ploutos Architecture)
  2. diagram_modified_method.png  (Ploutos+ Stage 2 Architecture with New Innovations Highlighted)
Outputs saved to results/figures/
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "data" / "interim")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

OUT_DIR = PROJECT_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_original_diagram():
    """Draws the Block Diagram for the Original Ploutos Method."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Title
    ax.text(0.5, 0.95, "BLOCK DIAGRAM: ORIGINAL PLOUTOS METHOD (Tong et al., 2024)",
            color='#38bdf8', fontsize=14, fontweight='bold', ha='center', va='top')
    ax.text(0.5, 0.90, "Pipeline: Multimodal Ingestion -> Three Independent Experts -> Static Equal Weighting -> LLM",
            color='#94a3b8', fontsize=10, ha='center', va='top')

    def add_box(x, y, w, h, title, subtitle, bg_color='#161b22', edge_color='#30363d', title_color='#ffffff'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor=bg_color, edgecolor=edge_color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.68, title, color=title_color, fontsize=9.5, fontweight='bold',
                ha='center', va='center')
        ax.text(x + w/2, y + h*0.32, subtitle, color='#94a3b8', fontsize=8,
                ha='center', va='center', multialignment='center')

    def add_arrow(x1, y1, x2, y2, color='#38bdf8'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0, mutation_scale=15))

    # Box 1: Input Data
    add_box(0.04, 0.40, 0.20, 0.38,
            "1. MULTIMODAL INPUTS",
            "• Daily OHLCV Prices (5 Days)\n• Social Cashtag Tweets (<=10/day)\n• Corporate Sector Category",
            bg_color='#162032', edge_color='#38bdf8', title_color='#38bdf8')

    # Three Parallel Expert Boxes
    add_box(0.30, 0.64, 0.20, 0.20,
            "Technical Expert",
            "5d Momentum, 14d RSI,\nRolling 5d Volatility",
            bg_color='#1e293b', edge_color='#64748b', title_color='#f1f5f9')

    add_box(0.30, 0.40, 0.20, 0.20,
            "Sentiment Expert",
            "VADER Tweet Polarity,\nPositive/Negative Cues",
            bg_color='#1e293b', edge_color='#64748b', title_color='#f1f5f9')

    add_box(0.30, 0.16, 0.20, 0.20,
            "Human Expert Proxy",
            "Sector Macro Profile,\nFundamental Demand",
            bg_color='#1e293b', edge_color='#64748b', title_color='#f1f5f9')

    # Box 3: Static Synthesizer
    add_box(0.56, 0.38, 0.18, 0.42,
            "3. STATIC PROMPT BUILDER",
            "Equal Fixed Weighting:\n• 33.3% Technical\n• 33.3% Sentiment\n• 33.4% Human Proxy\n\n[CRITICAL FLAW:\nCannot adapt to news gaps\nor market regime shifts]",
            bg_color='#2d151e', edge_color='#f43f5e', title_color='#fb7185')

    # Box 4: Base LLM Reasoner
    add_box(0.78, 0.50, 0.18, 0.25,
            "4. FINANCIAL LLM",
            "Instruction-Tuned\nBase Model (e.g., Llama/Qwen)\nGenerates Chain-of-Thought",
            bg_color='#162032', edge_color='#38bdf8', title_color='#38bdf8')

    # Box 5: Output
    add_box(0.78, 0.16, 0.18, 0.26,
            "5. OUTPUT PREDICTION",
            "• Direction: UP (1) / DOWN (0)\n• Rearview-Mirror Rationale\n(Qualitative justification)",
            bg_color='#06241b', edge_color='#10b981', title_color='#34d399')

    # Arrows
    add_arrow(0.24, 0.60, 0.30, 0.74, color='#38bdf8')
    add_arrow(0.24, 0.59, 0.30, 0.50, color='#38bdf8')
    add_arrow(0.24, 0.58, 0.30, 0.26, color='#38bdf8')

    add_arrow(0.50, 0.74, 0.56, 0.64, color='#64748b')
    add_arrow(0.50, 0.50, 0.56, 0.59, color='#64748b')
    add_arrow(0.50, 0.26, 0.56, 0.54, color='#64748b')

    add_arrow(0.74, 0.60, 0.78, 0.62, color='#fb7185')
    add_arrow(0.87, 0.50, 0.87, 0.42, color='#34d399')

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    plt.tight_layout()
    out_file = OUT_DIR / "diagram_original_method.png"
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Diagram] Generated: {out_file}")


def draw_modified_diagram():
    """Draws the Block Diagram for the Modified Ploutos+ Method."""
    fig, ax = plt.subplots(figsize=(13.333, 7.0), dpi=300)
    fig.patch.set_facecolor('#080c14')
    ax.set_facecolor('#080c14')

    # Title Banner
    ax.text(0.5, 0.96, "BLOCK DIAGRAM: PROPOSED MODIFIED METHOD (PLOUTOS+ ARCHITECTURE)",
            color='#10b981', fontsize=15, fontweight='bold', ha='center', va='top')
    ax.text(0.5, 0.91, "Novel Innovations Highlighted: 16-D Adaptive Neural Router  •  15-Day Candlestick Vision Expert  •  Actionable Risk Guardrails",
            color='#cbd5e1', fontsize=10, ha='center', va='top')

    def add_box(x, y, w, h, title, subtitle, bg_color='#0f172a', edge_color='#1e293b', title_color='#ffffff', is_new=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor=bg_color, edgecolor=edge_color, linewidth=2.0 if is_new else 1.2)
        ax.add_patch(box)
        
        # Badge for NEW components
        if is_new:
            badge = FancyBboxPatch((x + w - 0.055, y + h - 0.035), 0.05, 0.03, boxstyle="round,pad=0.005",
                                   facecolor='#10b981', edgecolor='none')
            ax.add_patch(badge)
            ax.text(x + w - 0.03, y + h - 0.02, "NEW", color='#000000', fontsize=6.5, fontweight='bold',
                    ha='center', va='center')

        ax.text(x + w/2, y + h*0.72, title, color=title_color, fontsize=9.2, fontweight='bold',
                ha='center', va='center')
        ax.text(x + w/2, y + h*0.35, subtitle, color='#94a3b8', fontsize=7.8,
                ha='center', va='center', multialignment='center')

    def add_arrow(x1, y1, x2, y2, color='#10b981'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0, mutation_scale=15))

    # Column 1: Input Modalities (Includes Candlestick Charts)
    add_box(0.03, 0.42, 0.17, 0.42,
            "1. MULTIMODAL INGESTION",
            "• OHLCV Trailing Prices (5d)\n• Cashtag Public Tweets\n• Corporate Sector Profiles\n• 15-Day Candlestick Images\n(Strictly lagged t < t+1)",
            bg_color='#0f1f2e', edge_color='#38bdf8', title_color='#38bdf8', is_new=True)

    # Column 2: 4 Parallel Feature Experts (Including Visual Expert)
    add_box(0.24, 0.69, 0.17, 0.17,
            "Technical Expert",
            "Wilder RSI, 5d Momentum,\nRolling Volatility, Volume Surge",
            bg_color='#1e293b', edge_color='#64748b', title_color='#f8fafc')

    add_box(0.24, 0.48, 0.17, 0.17,
            "Sentiment Expert",
            "10-Tweet Cap Polarity,\nLive Social Stream Headlines",
            bg_color='#1e293b', edge_color='#64748b', title_color='#f8fafc')

    add_box(0.24, 0.27, 0.17, 0.17,
            "Visual Expert",
            "Hammers, Engulfing Bars,\nMoving Average Geometry",
            bg_color='#062b24', edge_color='#10b981', title_color='#34d399', is_new=True)

    add_box(0.24, 0.06, 0.17, 0.17,
            "Human Proxy Expert",
            "Macro Sensitivity Profiles,\nSector Supply-Chain Factors",
            bg_color='#1e293b', edge_color='#64748b', title_color='#f8fafc')

    # Column 3: 16-D Feature Vector & Adaptive Neural Router (THE CORE INNOVATION)
    add_box(0.45, 0.42, 0.18, 0.44,
            "ADAPTIVE MOE ROUTER",
            "16-D Neural MLP Gating Network:\n[Volatility, Momentum, RSI,\nDispersion, Tweet Volume]\n\nComputes Softmax Weights:\nalpha_Tech: 42%\nalpha_Sent: 28%\nalpha_Fund: 30%\n\n*Survives Market Volatility Shocks!",
            bg_color='#06241b', edge_color='#10b981', title_color='#10b981', is_new=True)

    # Column 4: Fine-Tuned Local LLM Reasoner
    add_box(0.67, 0.42, 0.15, 0.44,
            "FINANCIAL LLM",
            "Qwen2.5-1.5B-Instruct\nFine-Tuned with PEFT LoRA\n(r=16, alpha=32 on RTX 4050)\n\nSynthesizes Grounded Prompt\nGenerates Direction + Rationale",
            bg_color='#1a1f36', edge_color='#818cf8', title_color='#a5b4fc', is_new=True)

    # Column 5: Actionable Guardrails & Real-Time Terminal Output
    add_box(0.85, 0.50, 0.13, 0.36,
            "ACTIONABLE RISK",
            "• Stop-Loss (1.5*sigma)\n• Take-Profit (2.5*sigma)\n• R:R Ratio (1:1.67)\n• Capital Sizing:\nSTRONG BUY / HOLD",
            bg_color='#2a1725', edge_color='#f43f5e', title_color='#fb7185', is_new=True)

    add_box(0.85, 0.08, 0.13, 0.36,
            "LIVE TERMINAL",
            "• TradingView Canvas\n• Forward Audit Ledger\n• Multi-Currency (INR / $)\n• 69/69 Tests Verified",
            bg_color='#0d2438', edge_color='#06b6d4', title_color='#38bdf8', is_new=True)

    # Arrows
    add_arrow(0.20, 0.63, 0.24, 0.77, color='#38bdf8')
    add_arrow(0.20, 0.63, 0.24, 0.56, color='#38bdf8')
    add_arrow(0.20, 0.63, 0.24, 0.35, color='#10b981')
    add_arrow(0.20, 0.63, 0.24, 0.14, color='#64748b')

    add_arrow(0.41, 0.77, 0.45, 0.68, color='#64748b')
    add_arrow(0.41, 0.56, 0.45, 0.64, color='#64748b')
    add_arrow(0.41, 0.35, 0.45, 0.60, color='#10b981')
    add_arrow(0.41, 0.14, 0.45, 0.56, color='#64748b')

    add_arrow(0.63, 0.64, 0.67, 0.64, color='#10b981')
    add_arrow(0.82, 0.68, 0.85, 0.68, color='#fb7185')
    add_arrow(0.82, 0.60, 0.85, 0.26, color='#06b6d4')

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    plt.tight_layout()
    out_file = OUT_DIR / "diagram_modified_method.png"
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Diagram] Generated: {out_file}")


if __name__ == "__main__":
    draw_original_diagram()
    draw_modified_diagram()
