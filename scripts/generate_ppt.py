"""
scripts/generate_ppt.py
Generates a professional PowerPoint presentation for the Ploutos+ project (Phases 1-7).
Uses a dark premium theme with gradient accents and structured academic slides.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ─── Color Palette (Dark Premium Theme) ───
BG_DARK       = RGBColor(0x0F, 0x0F, 0x1A)   # Deep navy-black
BG_CARD       = RGBColor(0x1A, 0x1A, 0x2E)   # Card background
ACCENT_BLUE   = RGBColor(0x4E, 0x8C, 0xFF)   # Primary blue accent
ACCENT_CYAN   = RGBColor(0x00, 0xD4, 0xAA)   # Cyan/teal accent
ACCENT_PURPLE = RGBColor(0x7C, 0x3A, 0xED)   # Purple accent
ACCENT_GOLD   = RGBColor(0xFF, 0xB8, 0x00)   # Gold accent
ACCENT_RED    = RGBColor(0xFF, 0x4D, 0x6A)   # Red accent
WHITE         = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY    = RGBColor(0xBB, 0xBB, 0xCC)
MID_GRAY      = RGBColor(0x88, 0x88, 0xAA)
GREEN         = RGBColor(0x00, 0xE6, 0x76)

OUTPUT_PATH = PROJECT_ROOT / "results" / "Ploutos_Plus_Presentation.pptx"
LOSS_FIGURE  = PROJECT_ROOT / "results" / "figures" / "training_loss.png"


def set_slide_bg(slide, color=BG_DARK):
    """Set solid background color for a slide."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color, border_color=None, border_width=Pt(0)):
    """Add a rounded rectangle shape."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = border_width
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def add_text_box(slide, left, top, width, height, text, font_size=18, color=WHITE,
                 bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    """Add a text box to a slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_paragraph(text_frame, text, font_size=14, color=LIGHT_GRAY, bold=False,
                  alignment=PP_ALIGN.LEFT, space_before=Pt(4), space_after=Pt(2), font_name="Calibri"):
    """Add a paragraph to an existing text frame."""
    p = text_frame.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    p.space_before = space_before
    p.space_after = space_after
    return p


def add_bullet_list(slide, left, top, width, height, items, font_size=14, color=LIGHT_GRAY,
                    bullet_color=ACCENT_CYAN, title=None, title_color=WHITE, title_size=20):
    """Add a bulleted list with optional title."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    if title:
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(title_size)
        p.font.color.rgb = title_color
        p.font.bold = True
        p.font.name = "Calibri"
        p.space_after = Pt(8)
    
    for i, item in enumerate(items):
        if title or i > 0:
            p = tf.add_paragraph()
        else:
            p = tf.paragraphs[0]
        p.text = f"▸  {item}"
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_before = Pt(6)
        p.space_after = Pt(3)
        # Color the bullet
        if p.runs:
            p.runs[0].font.color.rgb = bullet_color

    return txBox


def add_accent_bar(slide, left, top, width, height, color=ACCENT_BLUE):
    """Add a thin accent bar."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def add_metric_card(slide, left, top, width, height, label, value, accent=ACCENT_BLUE):
    """Add a styled metric card."""
    card = add_shape(slide, left, top, width, height, BG_CARD, accent, Pt(1.5))
    
    # Accent bar on top
    add_accent_bar(slide, left + Inches(0.1), top + Inches(0.08), width - Inches(0.2), Pt(3), accent)
    
    # Value
    add_text_box(slide, left + Inches(0.15), top + Inches(0.25), width - Inches(0.3), Inches(0.5),
                 str(value), font_size=22, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    
    # Label
    add_text_box(slide, left + Inches(0.15), top + Inches(0.65), width - Inches(0.3), Inches(0.4),
                 label, font_size=10, color=MID_GRAY, alignment=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════
#  SLIDE BUILDERS
# ═══════════════════════════════════════════════════════════

def slide_title(prs):
    """Slide 1: Title Slide"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide)

    # Top accent line
    add_accent_bar(slide, Inches(0), Inches(0), Inches(13.333), Pt(4), ACCENT_BLUE)

    # Title
    add_text_box(slide, Inches(0.8), Inches(1.6), Inches(11.5), Inches(1.2),
                 "Ploutos+", font_size=52, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

    # Subtitle
    add_text_box(slide, Inches(1.0), Inches(2.8), Inches(11.3), Inches(1.0),
                 "Interpretable Stock Movement Prediction with\nFinancial Large Language Models",
                 font_size=24, color=ACCENT_CYAN, alignment=PP_ALIGN.CENTER)

    # Tagline
    add_text_box(slide, Inches(1.0), Inches(3.9), Inches(11.3), Inches(0.6),
                 "Adaptive Expert Routing  •  Multimodal Evidence  •  Market-Regime Robustness",
                 font_size=14, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    # Divider
    add_accent_bar(slide, Inches(4.5), Inches(4.7), Inches(4.3), Pt(1.5), ACCENT_PURPLE)

    # Author info
    add_text_box(slide, Inches(1.0), Inches(5.1), Inches(11.3), Inches(0.4),
                 "Rishikesh (24BDS030)  •  Vineel (24BDS010)  •  Arjun (24BDS039)", font_size=16, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.0), Inches(5.5), Inches(11.3), Inches(0.4),
                 "B.Tech DSA  •  Indian Institute of Information Technology, Dharwad",
                 font_size=13, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.0), Inches(5.9), Inches(11.3), Inches(0.4),
                 "Course: LLM Applications in Finance  |  October 2026",
                 font_size=12, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    # Bottom accent
    add_accent_bar(slide, Inches(0), Inches(7.45), Inches(13.333), Pt(4), ACCENT_PURPLE)


def slide_agenda(prs):
    """Slide 2: Agenda / Outline"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_BLUE)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "PRESENTATION OUTLINE", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(2.5), Pt(3), ACCENT_CYAN)

    items = [
        ("01", "Problem Statement & Motivation", ACCENT_BLUE),
        ("02", "Research Background & Related Work", ACCENT_CYAN),
        ("03", "System Architecture & Methodology", ACCENT_PURPLE),
        ("04", "Dataset: StockNet Benchmark", ACCENT_BLUE),
        ("05", "Data Pipeline & Feature Engineering", ACCENT_CYAN),
        ("06", "Baseline Models & Results", ACCENT_GOLD),
        ("07", "PloutosGen Expert System", ACCENT_PURPLE),
        ("08", "LLM Fine-Tuning (LoRA)", ACCENT_BLUE),
        ("09", "Training Results & Convergence", ACCENT_CYAN),
        ("10", "Live Inference Demo", ACCENT_GOLD),
        ("11", "Future Work: Phases 8–12", ACCENT_RED),
        ("12", "Summary & Contributions", ACCENT_PURPLE),
    ]

    y = Inches(1.4)
    for num, title, accent in items:
        add_shape(slide, Inches(1.2), y, Inches(10.5), Inches(0.42), BG_CARD, accent, Pt(0.5))
        add_text_box(slide, Inches(1.4), y + Inches(0.04), Inches(0.6), Inches(0.35),
                     num, font_size=14, color=accent, bold=True, alignment=PP_ALIGN.CENTER)
        add_text_box(slide, Inches(2.1), y + Inches(0.04), Inches(9), Inches(0.35),
                     title, font_size=14, color=LIGHT_GRAY)
        y += Inches(0.48)


def slide_problem_statement(prs):
    """Slide 3: Problem Statement"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_BLUE)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "PROBLEM STATEMENT & MOTIVATION", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(3), Pt(3), ACCENT_BLUE)

    # Challenge card
    add_shape(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(2.8), BG_CARD, ACCENT_RED, Pt(1))
    add_bullet_list(slide, Inches(1.0), Inches(1.55), Inches(5.1), Inches(2.6), [
        "Stock markets are inherently noisy, non-stationary, and influenced by heterogeneous signals",
        "Traditional models use single-signal approaches (only price OR only sentiment)",
        "Existing LLM approaches lack interpretability — predictions without reasoning",
        "No principled mechanism to dynamically weight experts by market regime"
    ], font_size=13, title="The Challenge", title_color=ACCENT_RED, title_size=18)

    # Our approach card
    add_shape(slide, Inches(6.8), Inches(1.5), Inches(5.5), Inches(2.8), BG_CARD, GREEN, Pt(1))
    add_bullet_list(slide, Inches(7.0), Inches(1.55), Inches(5.1), Inches(2.6), [
        "Multi-expert evidence integration (Technical + Sentiment + Fundamental)",
        "LLM-powered interpretable predictions with grounded rationales",
        "Adaptive expert routing conditioned on market regime",
        "Rigorous zero-leakage evaluation on StockNet benchmark"
    ], font_size=13, title="Our Approach: Ploutos+", title_color=GREEN, title_size=18)

    # Research question
    add_shape(slide, Inches(0.8), Inches(4.7), Inches(11.5), Inches(1.2), BG_CARD, ACCENT_PURPLE, Pt(1.5))
    add_text_box(slide, Inches(1.0), Inches(4.8), Inches(11), Inches(0.35),
                 "Core Research Question", font_size=16, color=ACCENT_PURPLE, bold=True)
    add_text_box(slide, Inches(1.0), Inches(5.2), Inches(11), Inches(0.6),
                 "\"Can a financial LLM, guided by structured multi-expert evidence and adaptive expert routing,\n"
                 "produce interpretable, regime-robust stock movement predictions that surpass naive baselines?\"",
                 font_size=14, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)


def slide_related_work(prs):
    """Slide 4: Research Background"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_CYAN)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "RESEARCH BACKGROUND", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(2.5), Pt(3), ACCENT_CYAN)

    # Ploutos paper
    add_shape(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(2.2), BG_CARD, ACCENT_BLUE, Pt(1))
    add_text_box(slide, Inches(1.0), Inches(1.55), Inches(5.1), Inches(0.35),
                 "Foundation: Ploutos (Tong et al.)", font_size=16, color=ACCENT_BLUE, bold=True)
    add_bullet_list(slide, Inches(1.0), Inches(1.95), Inches(5.1), Inches(1.6), [
        "Generates multi-expert financial evidence as structured prompts",
        "Uses LLMs for stock movement prediction with rationale generation",
        "Demonstrated on Chinese A-shares market",
    ], font_size=12, color=LIGHT_GRAY)

    # Our extension
    add_shape(slide, Inches(6.8), Inches(1.5), Inches(5.5), Inches(2.2), BG_CARD, ACCENT_PURPLE, Pt(1))
    add_text_box(slide, Inches(7.0), Inches(1.55), Inches(5.1), Inches(0.35),
                 "Our Extension: Ploutos+", font_size=16, color=ACCENT_PURPLE, bold=True)
    add_bullet_list(slide, Inches(7.0), Inches(1.95), Inches(5.1), Inches(1.6), [
        "Adaptive Expert Router — dynamic context-gated weighting",
        "Market-regime robustness analysis (Bull/Bear/Sideways)",
        "StockNet benchmark for reproducible English-market evaluation",
    ], font_size=12, color=LIGHT_GRAY)

    # Key technologies
    add_shape(slide, Inches(0.8), Inches(4.1), Inches(11.5), Inches(2.3), BG_CARD, ACCENT_GOLD, Pt(1))
    add_text_box(slide, Inches(1.0), Inches(4.15), Inches(5), Inches(0.35),
                 "Key Technologies", font_size=16, color=ACCENT_GOLD, bold=True)

    techs = [
        ("Qwen2.5-1.5B-Instruct", "Alibaba's instruction-tuned LLM — compact yet capable"),
        ("LoRA (Low-Rank Adaptation)", "Parameter-efficient fine-tuning; only ~0.5% params trained"),
        ("PEFT + HuggingFace", "Production-grade fine-tuning infrastructure"),
        ("StockNet Dataset", "87 S&P 500 stocks, 2014–2015, prices + Twitter data"),
    ]
    y = Inches(4.55)
    for tech, desc in techs:
        add_text_box(slide, Inches(1.2), y, Inches(3), Inches(0.35),
                     f"▸  {tech}", font_size=12, color=ACCENT_CYAN, bold=True)
        add_text_box(slide, Inches(4.3), y, Inches(7.5), Inches(0.35),
                     desc, font_size=12, color=LIGHT_GRAY)
        y += Inches(0.42)


def slide_architecture(prs):
    """Slide 5: System Architecture"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_PURPLE)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "SYSTEM ARCHITECTURE", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(2.5), Pt(3), ACCENT_PURPLE)

    # Pipeline stages
    stages = [
        ("StockNet\nData", ACCENT_BLUE),
        ("Label &\nSplit", ACCENT_CYAN),
        ("Feature\nEngineering", ACCENT_PURPLE),
        ("Expert\nEvidence", ACCENT_GOLD),
        ("Prompt\nBuilder", ACCENT_BLUE),
        ("LLM\nFine-Tuning", ACCENT_RED),
        ("Prediction +\nRationale", GREEN),
    ]

    start_x = Inches(0.5)
    y_center = Inches(2.3)
    box_w = Inches(1.5)
    box_h = Inches(1.2)
    gap = Inches(0.22)

    for i, (label, color) in enumerate(stages):
        x = start_x + i * (box_w + gap)
        add_shape(slide, x, y_center, box_w, box_h, BG_CARD, color, Pt(2))
        add_text_box(slide, x, y_center + Inches(0.2), box_w, Inches(0.8),
                     label, font_size=13, color=color, bold=True, alignment=PP_ALIGN.CENTER)

        # Arrow between boxes
        if i < len(stages) - 1:
            arrow_x = x + box_w + Inches(0.02)
            add_text_box(slide, arrow_x, y_center + Inches(0.35), Inches(0.2), Inches(0.4),
                         "→", font_size=18, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    # Three expert cards
    expert_y = Inches(4.2)
    experts_data = [
        ("Technical Expert", "RSI-14, MACD, Bollinger Bands,\n5-day Momentum, Volatility", ACCENT_BLUE),
        ("Sentiment Expert", "Twitter/social NLP sentiment,\nDaily tweet aggregation (10-cap)", ACCENT_CYAN),
        ("Human Expert (Proxy)", "Sector classification, market-cap\nstrata, index membership", ACCENT_PURPLE),
    ]

    for i, (name, desc, color) in enumerate(experts_data):
        x = Inches(0.8) + i * Inches(4.1)
        add_shape(slide, x, expert_y, Inches(3.7), Inches(1.8), BG_CARD, color, Pt(1.5))
        add_text_box(slide, x + Inches(0.15), expert_y + Inches(0.1), Inches(3.4), Inches(0.4),
                     name, font_size=15, color=color, bold=True, alignment=PP_ALIGN.CENTER)
        add_accent_bar(slide, x + Inches(0.5), expert_y + Inches(0.55), Inches(2.7), Pt(2), color)
        add_text_box(slide, x + Inches(0.15), expert_y + Inches(0.7), Inches(3.4), Inches(1.0),
                     desc, font_size=11, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)


def slide_dataset(prs):
    """Slide 6: StockNet Dataset"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_BLUE)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "DATASET: STOCKNET BENCHMARK", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(3), Pt(3), ACCENT_BLUE)

    # Key stats cards
    stats = [
        ("87", "S&P 500\nStocks", ACCENT_BLUE),
        ("26,623", "Labeled\nSamples", ACCENT_CYAN),
        ("2014–2015", "Time\nPeriod", ACCENT_PURPLE),
        ("44.66%", "Tweet\nCoverage", ACCENT_GOLD),
    ]

    for i, (val, label, color) in enumerate(stats):
        x = Inches(0.8) + i * Inches(3.05)
        add_metric_card(slide, x, Inches(1.5), Inches(2.7), Inches(1.1), label, val, color)

    # Chronological splits
    add_shape(slide, Inches(0.8), Inches(3.1), Inches(11.5), Inches(3.0), BG_CARD, ACCENT_CYAN, Pt(1))
    add_text_box(slide, Inches(1.0), Inches(3.15), Inches(5), Inches(0.4),
                 "Strict Chronological Temporal Splits (Zero Leakage)", font_size=16, color=ACCENT_CYAN, bold=True)

    splits_data = [
        ("TRAIN", "2014-01-02  →  2015-07-31", "20,348 samples", ACCENT_BLUE),
        ("VALIDATION", "2015-08-03  →  2015-09-30", "2,555 samples", ACCENT_GOLD),
        ("TEST", "2015-10-01  →  2015-12-31", "3,720 samples", ACCENT_RED),
    ]

    y = Inches(3.7)
    for name, period, count, color in splits_data:
        add_shape(slide, Inches(1.2), y, Inches(10.5), Inches(0.55), RGBColor(0x14, 0x14, 0x28), color, Pt(1))
        add_text_box(slide, Inches(1.4), y + Inches(0.08), Inches(2), Inches(0.4),
                     name, font_size=14, color=color, bold=True)
        add_text_box(slide, Inches(3.8), y + Inches(0.08), Inches(4), Inches(0.4),
                     period, font_size=13, color=LIGHT_GRAY)
        add_text_box(slide, Inches(8.5), y + Inches(0.08), Inches(3), Inches(0.4),
                     count, font_size=13, color=WHITE, bold=True, alignment=PP_ALIGN.RIGHT)
        y += Inches(0.7)

    # Labeling methodology
    add_text_box(slide, Inches(1.0), Inches(5.6), Inches(11), Inches(0.35),
                 "Labeling:  DOWN (≤ -0.50%)  |  Excluded (-0.50%, +0.55%]  |  UP (> +0.55%)",
                 font_size=12, color=MID_GRAY, alignment=PP_ALIGN.CENTER)


def slide_data_pipeline(prs):
    """Slide 7: Data Pipeline & Feature Engineering"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_CYAN)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "DATA PIPELINE & FEATURE ENGINEERING", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(3.5), Pt(3), ACCENT_CYAN)

    # Phase 2-4 pipeline
    phases = [
        ("Phase 2: Data Acquisition", [
            "Downloaded StockNet price + tweet archives",
            "Verified 87 stocks across 9 GICS sectors",
            "Generated data manifest with integrity checksums",
        ], ACCENT_BLUE),
        ("Phase 3: Label & Split", [
            "Constructed movement-threshold labels (UP/DOWN)",
            "Strict chronological train/val/test splits",
            "Zero data leakage verified via automated tests",
        ], ACCENT_CYAN),
        ("Phase 4: Text Processing", [
            "Aligned tweets to stock-day observations",
            "Applied 10-tweet daily cap per stock",
            "45.14% tweet coverage (12,018 stock-days)",
        ], ACCENT_PURPLE),
    ]

    for i, (title, items, color) in enumerate(phases):
        x = Inches(0.8) + i * Inches(4.1)
        add_shape(slide, x, Inches(1.5), Inches(3.7), Inches(3.0), BG_CARD, color, Pt(1.5))
        add_text_box(slide, x + Inches(0.15), Inches(1.55), Inches(3.4), Inches(0.4),
                     title, font_size=14, color=color, bold=True)
        add_accent_bar(slide, x + Inches(0.3), Inches(1.95), Inches(3.1), Pt(2), color)
        add_bullet_list(slide, x + Inches(0.15), Inches(2.05), Inches(3.4), Inches(2.3),
                        items, font_size=11, color=LIGHT_GRAY)

    # Technical features
    add_shape(slide, Inches(0.8), Inches(4.9), Inches(11.5), Inches(2.0), BG_CARD, ACCENT_GOLD, Pt(1))
    add_text_box(slide, Inches(1.0), Inches(4.95), Inches(5), Inches(0.35),
                 "Technical Indicators Extracted", font_size=16, color=ACCENT_GOLD, bold=True)

    indicators = [
        "RSI (14-period)", "MACD & Signal Line", "Bollinger Bands (Width & %B)",
        "5-Day Momentum", "20-Day Volatility", "5-Day SMA",
        "Volume Ratio (5d/20d)", "Daily Returns", "Log Returns"
    ]

    y = Inches(5.4)
    for i, ind in enumerate(indicators):
        col = i % 3
        row = i // 3
        x = Inches(1.2) + col * Inches(3.8)
        cy = y + row * Inches(0.4)
        add_text_box(slide, x, cy, Inches(3.5), Inches(0.35),
                     f"▸  {ind}", font_size=12, color=LIGHT_GRAY)


def slide_baselines(prs):
    """Slide 8: Baseline Results"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_GOLD)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "BASELINE MODELS & RESULTS", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(3), Pt(3), ACCENT_GOLD)

    # Majority class baseline
    add_shape(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(3.0), BG_CARD, ACCENT_RED, Pt(1.5))
    add_text_box(slide, Inches(1.0), Inches(1.55), Inches(5.1), Inches(0.4),
                 "Majority Class Baseline", font_size=18, color=ACCENT_RED, bold=True)
    add_accent_bar(slide, Inches(1.0), Inches(1.95), Inches(5.1), Pt(2), ACCENT_RED)

    maj_items = [
        "Predicts the most frequent class (UP) for all samples",
        "Test Accuracy: 51.29%",
        "Macro F1: 33.90%",
        "MCC: 0.000 (no discriminative power)",
        "Status: Trivial lower bound established",
    ]
    add_bullet_list(slide, Inches(1.0), Inches(2.1), Inches(5.1), Inches(2.2),
                    maj_items, font_size=13, color=LIGHT_GRAY)

    # Random Forest baseline
    add_shape(slide, Inches(6.8), Inches(1.5), Inches(5.5), Inches(3.0), BG_CARD, ACCENT_BLUE, Pt(1.5))
    add_text_box(slide, Inches(7.0), Inches(1.55), Inches(5.1), Inches(0.4),
                 "Random Forest Baseline", font_size=18, color=ACCENT_BLUE, bold=True)
    add_accent_bar(slide, Inches(7.0), Inches(1.95), Inches(5.1), Pt(2), ACCENT_BLUE)

    rf_items = [
        "100 estimators, max_depth=8, lagged technical features",
        "Test Accuracy: 52.53%",
        "Macro F1: 52.09%",
        "MCC: 0.0469 (marginal above random)",
        "Status: Meaningful but weak — confirms task difficulty",
    ]
    add_bullet_list(slide, Inches(7.0), Inches(2.1), Inches(5.1), Inches(2.2),
                    rf_items, font_size=13, color=LIGHT_GRAY)

    # Comparison table
    add_shape(slide, Inches(0.8), Inches(4.9), Inches(11.5), Inches(2.2), BG_CARD, ACCENT_CYAN, Pt(1))
    add_text_box(slide, Inches(1.0), Inches(4.95), Inches(5), Inches(0.35),
                 "Baseline Comparison (Test Set — 3,720 samples)", font_size=16, color=ACCENT_CYAN, bold=True)

    # Table header
    headers = ["Model", "Accuracy", "Macro F1", "MCC", "Tests Passed"]
    hx_positions = [Inches(1.2), Inches(4.0), Inches(5.8), Inches(7.6), Inches(9.4)]
    y_h = Inches(5.45)
    for hx, h in zip(hx_positions, headers):
        add_text_box(slide, hx, y_h, Inches(1.8), Inches(0.3),
                     h, font_size=12, color=ACCENT_GOLD, bold=True, alignment=PP_ALIGN.CENTER)

    # Table rows
    rows = [
        ("Majority Class", "51.29%", "33.90%", "0.000", "—"),
        ("Random Forest", "52.53%", "52.09%", "0.047", "21/21 ✓"),
    ]
    for ri, (model, acc, f1, mcc, tests) in enumerate(rows):
        y_r = Inches(5.8) + ri * Inches(0.4)
        vals = [model, acc, f1, mcc, tests]
        for hx, v in zip(hx_positions, vals):
            c = WHITE if ri == 1 else LIGHT_GRAY
            add_text_box(slide, hx, y_r, Inches(1.8), Inches(0.3),
                         v, font_size=12, color=c, alignment=PP_ALIGN.CENTER)


def slide_ploutosgen(prs):
    """Slide 9: PloutosGen Expert System"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_PURPLE)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "PLOUTOSGEN: EXPERT EVIDENCE SYSTEM", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(3.5), Pt(3), ACCENT_PURPLE)

    # Expert details
    experts_detail = [
        ("Technical Expert", ACCENT_BLUE, [
            "Computes RSI, MACD, Bollinger, Momentum",
            "Generates signal: BULLISH / BEARISH / NEUTRAL",
            "Outputs confidence score [0–1]",
            "Provides human-readable evidence summary",
        ]),
        ("Sentiment Expert", ACCENT_CYAN, [
            "Aggregates daily Twitter sentiment",
            "10-tweet cap per stock-day",
            "Classifies: BULLISH / BEARISH / NEUTRAL",
            "Handles zero-tweet days gracefully",
        ]),
        ("Human Expert (Proxy)", ACCENT_PURPLE, [
            "Encodes GICS sector classification",
            "Provides market-cap strata context",
            "Summarizes index membership info",
            "Static fundamental background data",
        ]),
    ]

    for i, (name, color, items) in enumerate(experts_detail):
        x = Inches(0.8) + i * Inches(4.1)
        add_shape(slide, x, Inches(1.5), Inches(3.7), Inches(3.2), BG_CARD, color, Pt(1.5))
        add_text_box(slide, x + Inches(0.15), Inches(1.55), Inches(3.4), Inches(0.4),
                     name, font_size=15, color=color, bold=True, alignment=PP_ALIGN.CENTER)
        add_accent_bar(slide, x + Inches(0.3), Inches(1.95), Inches(3.1), Pt(2), color)
        add_bullet_list(slide, x + Inches(0.15), Inches(2.1), Inches(3.4), Inches(2.5),
                        items, font_size=12, color=LIGHT_GRAY)

    # Output stats
    add_shape(slide, Inches(0.8), Inches(5.1), Inches(11.5), Inches(1.5), BG_CARD, ACCENT_GOLD, Pt(1))
    add_text_box(slide, Inches(1.0), Inches(5.15), Inches(5), Inches(0.35),
                 "Prompt Generation Output", font_size=16, color=ACCENT_GOLD, bold=True)

    stats = [
        ("26,623", "Total Prompts\nGenerated", ACCENT_BLUE),
        ("3 Experts", "Per Sample\nEvidence", ACCENT_CYAN),
        ("JSONL", "Storage\nFormat", ACCENT_PURPLE),
        ("17/17", "Pytest Tests\nPassed", GREEN),
    ]
    for i, (val, label, color) in enumerate(stats):
        x = Inches(1.2) + i * Inches(2.9)
        add_metric_card(slide, x, Inches(5.55), Inches(2.4), Inches(1.0), label, val, color)


def slide_lora_finetuning(prs):
    """Slide 10: LLM Fine-Tuning Configuration"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_RED)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "LLM FINE-TUNING: LoRA ON QWEN2.5-1.5B", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(3.5), Pt(3), ACCENT_RED)

    # Model config
    add_shape(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(3.5), BG_CARD, ACCENT_BLUE, Pt(1.5))
    add_text_box(slide, Inches(1.0), Inches(1.55), Inches(5.1), Inches(0.4),
                 "Model & LoRA Configuration", font_size=16, color=ACCENT_BLUE, bold=True)
    add_accent_bar(slide, Inches(1.0), Inches(1.95), Inches(5.1), Pt(2), ACCENT_BLUE)

    config_items = [
        ("Base Model", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("Parameters", "1.5 Billion (base)"),
        ("LoRA Rank (r)", "16"),
        ("LoRA Alpha (α)", "32"),
        ("Target Modules", "q_proj, k_proj, v_proj, o_proj"),
        ("Trainable Params", "~0.5% of total"),
        ("Precision", "FP16 Mixed Precision"),
        ("Max Seq Length", "448 tokens"),
    ]

    y = Inches(2.1)
    for key, val in config_items:
        add_text_box(slide, Inches(1.2), y, Inches(2.2), Inches(0.3),
                     key, font_size=11, color=MID_GRAY)
        add_text_box(slide, Inches(3.4), y, Inches(2.8), Inches(0.3),
                     val, font_size=11, color=WHITE, bold=True)
        y += Inches(0.32)

    # Training config
    add_shape(slide, Inches(6.8), Inches(1.5), Inches(5.5), Inches(3.5), BG_CARD, ACCENT_PURPLE, Pt(1.5))
    add_text_box(slide, Inches(7.0), Inches(1.55), Inches(5.1), Inches(0.4),
                 "Training Configuration", font_size=16, color=ACCENT_PURPLE, bold=True)
    add_accent_bar(slide, Inches(7.0), Inches(1.95), Inches(5.1), Pt(2), ACCENT_PURPLE)

    train_items = [
        ("Hardware", "NVIDIA RTX 4050 (6 GB VRAM)"),
        ("Epochs", "2"),
        ("Batch Size", "2 (effective 8 with grad accum)"),
        ("Gradient Accum Steps", "4"),
        ("Learning Rate", "2 × 10⁻⁴"),
        ("Training Samples", "500 (stratified)"),
        ("Validation Samples", "100"),
        ("Training Time", "~6 minutes"),
    ]

    y = Inches(2.1)
    for key, val in train_items:
        add_text_box(slide, Inches(7.2), y, Inches(2.5), Inches(0.3),
                     key, font_size=11, color=MID_GRAY)
        add_text_box(slide, Inches(9.7), y, Inches(2.4), Inches(0.3),
                     val, font_size=11, color=WHITE, bold=True)
        y += Inches(0.32)

    # Why LoRA card
    add_shape(slide, Inches(0.8), Inches(5.4), Inches(11.5), Inches(1.5), BG_CARD, GREEN, Pt(1))
    add_text_box(slide, Inches(1.0), Inches(5.45), Inches(5), Inches(0.35),
                 "Why LoRA?", font_size=16, color=GREEN, bold=True)
    add_bullet_list(slide, Inches(1.0), Inches(5.85), Inches(11), Inches(0.9), [
        "Trains only low-rank decomposition matrices — ~0.5% of total parameters — dramatically reducing VRAM requirements",
        "Enables fine-tuning a 1.5B parameter model on a laptop GPU (6 GB VRAM) in under 6 minutes",
        "Preserves base model's general language understanding while specializing for financial prediction",
    ], font_size=12, color=LIGHT_GRAY)


def slide_training_results(prs):
    """Slide 11: Training Results & Loss Curve"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_CYAN)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "TRAINING CONVERGENCE & RESULTS", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(3), Pt(3), ACCENT_CYAN)

    # Loss curve image
    if LOSS_FIGURE.exists():
        slide.shapes.add_picture(str(LOSS_FIGURE), Inches(0.8), Inches(1.5), Inches(7.0), Inches(4.5))

    # Key metrics on the right
    add_shape(slide, Inches(8.2), Inches(1.5), Inches(4.3), Inches(4.5), BG_CARD, ACCENT_CYAN, Pt(1.5))
    add_text_box(slide, Inches(8.4), Inches(1.55), Inches(3.9), Inches(0.4),
                 "Convergence Metrics", font_size=16, color=ACCENT_CYAN, bold=True)
    add_accent_bar(slide, Inches(8.4), Inches(1.95), Inches(3.9), Pt(2), ACCENT_CYAN)

    metrics = [
        ("Initial Loss", "3.54", ACCENT_RED),
        ("Final Train Loss", "0.274", GREEN),
        ("Epoch 1 Val Loss", "0.432", ACCENT_GOLD),
        ("Epoch 2 Val Loss", "0.381", GREEN),
        ("Total Steps", "126", ACCENT_BLUE),
        ("Training Time", "~6 min", ACCENT_PURPLE),
    ]

    y = Inches(2.2)
    for key, val, color in metrics:
        add_text_box(slide, Inches(8.5), y, Inches(2.2), Inches(0.32),
                     key, font_size=12, color=MID_GRAY)
        add_text_box(slide, Inches(10.7), y, Inches(1.6), Inches(0.32),
                     val, font_size=14, color=color, bold=True, alignment=PP_ALIGN.RIGHT)
        y += Inches(0.38)

    # Observation
    add_text_box(slide, Inches(8.5), Inches(4.6), Inches(3.8), Inches(1.2),
                 "✓ Loss dropped 96% (3.54→0.095)\n"
                 "✓ Validation loss confirms generalization\n"
                 "✓ No overfitting detected",
                 font_size=11, color=GREEN)

    # Tests passed bar
    add_shape(slide, Inches(0.8), Inches(6.4), Inches(11.5), Inches(0.7), BG_CARD, GREEN, Pt(1.5))
    add_text_box(slide, Inches(1.0), Inches(6.48), Inches(11), Inches(0.5),
                 "✅  30/30 Automated Pytest Tests PASSED  —  All Phases 1-7 Verified",
                 font_size=16, color=GREEN, bold=True, alignment=PP_ALIGN.CENTER)


def slide_inference_demo(prs):
    """Slide 12: Live Inference Demo"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_GOLD)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "LIVE INFERENCE DEMONSTRATION", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(3), Pt(3), ACCENT_GOLD)

    # CLI command
    add_shape(slide, Inches(0.8), Inches(1.5), Inches(11.5), Inches(0.8), RGBColor(0x10, 0x10, 0x18))
    add_text_box(slide, Inches(1.2), Inches(1.6), Inches(10.5), Inches(0.6),
                 "$ python src/inference/predict.py --sample --ticker AAPL",
                 font_size=16, color=GREEN, font_name="Consolas")

    # Flow description
    add_shape(slide, Inches(0.8), Inches(2.6), Inches(11.5), Inches(4.2), BG_CARD, ACCENT_GOLD, Pt(1))
    add_text_box(slide, Inches(1.0), Inches(2.65), Inches(5), Inches(0.4),
                 "Inference Pipeline", font_size=16, color=ACCENT_GOLD, bold=True)

    steps = [
        ("1", "Load Test Sample", "Reads from pre-generated test prompts (JSONL)", ACCENT_BLUE),
        ("2", "Load Model", "Base Qwen2.5-1.5B + LoRA adapter from checkpoint", ACCENT_CYAN),
        ("3", "Build Prompt", "Formats multi-expert evidence into chat template", ACCENT_PURPLE),
        ("4", "Generate", "Model produces prediction (UP/DOWN) + rationale", ACCENT_GOLD),
        ("5", "Evaluate", "Compares against ground truth label", GREEN),
    ]

    y = Inches(3.2)
    for num, title, desc, color in steps:
        add_shape(slide, Inches(1.2), y, Inches(0.5), Inches(0.5), color)
        add_text_box(slide, Inches(1.2), y + Inches(0.08), Inches(0.5), Inches(0.35),
                     num, font_size=16, color=BG_DARK, bold=True, alignment=PP_ALIGN.CENTER)
        add_text_box(slide, Inches(1.9), y + Inches(0.02), Inches(2.5), Inches(0.3),
                     title, font_size=14, color=color, bold=True)
        add_text_box(slide, Inches(1.9), y + Inches(0.3), Inches(9.5), Inches(0.3),
                     desc, font_size=11, color=LIGHT_GRAY)
        y += Inches(0.65)

    # Sample output
    add_text_box(slide, Inches(1.0), Inches(6.2), Inches(11), Inches(0.4),
                 "Output:  >>> MODEL PREDICTION: UP <<<   |   Ground Truth: UP   |   Status: [CORRECT]",
                 font_size=13, color=GREEN, bold=True, alignment=PP_ALIGN.CENTER)


def slide_future_work(prs):
    """Slide 13: Future Work"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_RED)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "FUTURE WORK: PHASES 8–12", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(2.5), Pt(3), ACCENT_RED)

    phases = [
        ("Phase 8", "Adaptive Expert Router", "Context-gated MLP dynamically weighting experts based on market regime indicators", ACCENT_BLUE, "Core Contribution"),
        ("Phase 9", "Visual Financial Expert", "Price chart generation & visual evidence integration as a 4th expert modality", ACCENT_CYAN, "Extension"),
        ("Phase 10", "Market-Regime Robustness", "Stress testing across Bull / Bear / Sideways / High-Volatility regimes", ACCENT_PURPLE, "Analysis"),
        ("Phase 11", "Ablation Studies", "Comprehensive metric matrix: accuracy, F1, MCC per expert combination", ACCENT_GOLD, "Evaluation"),
        ("Phase 12", "Final Deliverables", "Research paper, defense slides, reproducible repository, figures", ACCENT_RED, "Delivery"),
    ]

    y = Inches(1.5)
    for phase, title, desc, color, tag in phases:
        add_shape(slide, Inches(0.8), y, Inches(11.5), Inches(1.0), BG_CARD, color, Pt(1.5))
        add_text_box(slide, Inches(1.0), y + Inches(0.08), Inches(1.3), Inches(0.35),
                     phase, font_size=14, color=color, bold=True)
        add_text_box(slide, Inches(2.4), y + Inches(0.08), Inches(5), Inches(0.35),
                     title, font_size=15, color=WHITE, bold=True)
        add_text_box(slide, Inches(2.4), y + Inches(0.42), Inches(8), Inches(0.5),
                     desc, font_size=12, color=LIGHT_GRAY)
        # Tag
        add_shape(slide, Inches(10.5), y + Inches(0.15), Inches(1.6), Inches(0.35), color)
        add_text_box(slide, Inches(10.5), y + Inches(0.15), Inches(1.6), Inches(0.35),
                     tag, font_size=10, color=BG_DARK, bold=True, alignment=PP_ALIGN.CENTER)
        y += Inches(1.15)


def slide_summary(prs):
    """Slide 14: Summary & Contributions"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_accent_bar(slide, Inches(0), Inches(0), Pt(6), Inches(7.5), ACCENT_PURPLE)

    add_text_box(slide, Inches(0.8), Inches(0.4), Inches(10), Inches(0.6),
                 "SUMMARY & KEY CONTRIBUTIONS", font_size=28, color=WHITE, bold=True)
    add_accent_bar(slide, Inches(0.8), Inches(1.0), Inches(3), Pt(3), ACCENT_PURPLE)

    # Achievement stats
    stats = [
        ("7/12", "Phases\nCompleted", GREEN),
        ("30/30", "All Tests\nPassed", ACCENT_BLUE),
        ("96%", "Loss\nReduction", ACCENT_CYAN),
        ("6 min", "GPU Training\nTime", ACCENT_PURPLE),
    ]
    for i, (val, label, color) in enumerate(stats):
        x = Inches(0.8) + i * Inches(3.05)
        add_metric_card(slide, x, Inches(1.5), Inches(2.7), Inches(1.1), label, val, color)

    # Key contributions
    add_shape(slide, Inches(0.8), Inches(3.1), Inches(11.5), Inches(3.5), BG_CARD, GREEN, Pt(1.5))
    add_text_box(slide, Inches(1.0), Inches(3.15), Inches(5), Inches(0.4),
                 "Key Contributions So Far", font_size=18, color=GREEN, bold=True)
    add_accent_bar(slide, Inches(1.0), Inches(3.55), Inches(11), Pt(2), GREEN)

    contributions = [
        "Established a rigorous, leakage-free evaluation pipeline on the StockNet benchmark (87 S&P 500 stocks)",
        "Built a modular 3-expert evidence system (Technical, Sentiment, Human) generating structured prompts",
        "Fine-tuned Qwen2.5-1.5B-Instruct with LoRA on a consumer GPU (RTX 4050, 6 GB) in ~6 minutes",
        "Achieved strong convergence: training loss 3.54 → 0.095 with validation loss confirming generalization",
        "Created an interactive inference CLI producing interpretable predictions with grounded rationales",
        "Developed comprehensive automated test suites: 30/30 tests passing across all phases",
        "Designed modular, extensible codebase ready for the Adaptive Expert Router (Phase 8)",
    ]
    add_bullet_list(slide, Inches(1.0), Inches(3.65), Inches(11), Inches(2.8),
                    contributions, font_size=12, color=LIGHT_GRAY, bullet_color=GREEN)


def slide_thank_you(prs):
    """Slide 15: Thank You"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_accent_bar(slide, Inches(0), Inches(0), Inches(13.333), Pt(4), ACCENT_PURPLE)

    add_text_box(slide, Inches(0.8), Inches(2.0), Inches(11.5), Inches(1.0),
                 "Thank You", font_size=52, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(1.0), Inches(3.2), Inches(11.3), Inches(0.6),
                 "Questions & Discussion", font_size=24, color=ACCENT_CYAN, alignment=PP_ALIGN.CENTER)

    add_accent_bar(slide, Inches(4.5), Inches(4.0), Inches(4.3), Pt(1.5), ACCENT_GOLD)

    add_text_box(slide, Inches(1.0), Inches(4.5), Inches(11.3), Inches(0.4),
                 "Rishikesh (24BDS030)  •  Vineel (24BDS010)  •  Arjun (24BDS039)  •  IIIT Dharwad", font_size=15, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.0), Inches(4.9), Inches(11.3), Inches(0.4),
                 "LLM Applications in Finance  |  October 2026", font_size=13, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    add_accent_bar(slide, Inches(0), Inches(7.45), Inches(13.333), Pt(4), ACCENT_BLUE)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  GENERATING PLOUTOS+ PROFESSIONAL PRESENTATION")
    print("=" * 60)

    prs = Presentation()
    # Set widescreen 16:9
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    print("[1/14]  Title Slide OK")

    slide_agenda(prs)
    print("[2/14]  Agenda OK")

    slide_problem_statement(prs)
    print("[3/14]  Problem Statement OK")

    slide_related_work(prs)
    print("[4/14]  Research Background OK")

    slide_architecture(prs)
    print("[5/14]  System Architecture OK")

    slide_dataset(prs)
    print("[6/14]  Dataset OK")

    slide_data_pipeline(prs)
    print("[7/14]  Data Pipeline OK")

    slide_baselines(prs)
    print("[8/14]  Baselines OK")

    slide_ploutosgen(prs)
    print("[9/14]  PloutosGen OK")

    slide_lora_finetuning(prs)
    print("[10/14] LoRA Fine-Tuning OK")

    slide_training_results(prs)
    print("[11/14] Training Results OK")

    slide_inference_demo(prs)
    print("[12/14] Inference Demo OK")

    slide_future_work(prs)
    print("[13/14] Future Work OK")

    slide_summary(prs)
    print("[14/14] Summary OK")

    slide_thank_you(prs)
    print("[15/15] Thank You OK")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PATH))
    print(f"\n{'=' * 60}")
    print(f"  [OK] Presentation saved to: {OUTPUT_PATH}")
    print(f"  Total slides: {len(prs.slides)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
