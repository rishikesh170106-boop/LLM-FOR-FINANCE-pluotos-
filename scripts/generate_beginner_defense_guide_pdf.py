"""
scripts/generate_beginner_defense_guide_pdf.py
Generates the Ultimate Beginner-to-Expert Presentation Defense Guide PDF.
Zero finance knowledge required. Explains every UI element, abbreviation,
financial formula, presentation speaking script, and professor Q&A.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = PROJECT_ROOT / "results" / "reports" / "PLOUTOS_PLUS_BEGINNER_TO_EXPERT_DEFENSE_GUIDE.pdf"


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "PLOUTOS+ | Comprehensive Student Defense & UI Element-by-Element Master Guide")
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "IIIT Dharwad • LLM Applications in Finance")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Running Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * inch - 54, 46)

        self.setFont("Helvetica", 8)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawString(54, 32, "Confidential Student Presentation Master Dossier — Authors: Rishikesh (030), Vineel (010), Arjun (039)")
        self.drawRightString(8.5 * inch - 54, 32, page_str)
        self.restoreState()


def build_guide_pdf():
    PDF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0D9488"),
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=4
    )

    script_box_style = ParagraphStyle(
        "ScriptBox",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor("#065F46")
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=colors.white,
        alignment=1
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor("#0F172A")
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold"
    )

    table_cell_mono = ParagraphStyle(
        "TableCellMono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # Title Banner
    story.append(Paragraph("PLOUTOS+: Master Defense Dossier & Complete UI Element-by-Element Guide", title_style))
    story.append(Paragraph("Zero-Knowledge Presentation Manual: Stock Market Concepts, UI Widgets, Abbreviations & Word-for-Word Speaking Scripts", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D9488"), spaceAfter=6))

    # Meta box
    meta_data = [
        [
            Paragraph("<b>Institution:</b> IIIT Dharwad | <b>Course:</b> LLM Applications in Finance", body_style),
            Paragraph("<b>Team:</b> Rishikesh (24BDS030) • Vineel (24BDS010) • Arjun (24BDS039)", body_style)
        ],
        [
            Paragraph("<b>Live Terminal URL:</b> http://127.0.0.1:8000 (FastAPI + TradingView)", body_style),
            Paragraph("<b>Target Audience:</b> Any teammate can present this directly without prior finance knowledge.", body_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[250, 266])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    # =========================================================================
    # PART 1: STOCK MARKET 101 (ZERO PRIOR KNOWLEDGE)
    # =========================================================================
    story.append(Paragraph("PART 1: Stock Market 101 — Fundamental Concepts Explained from Scratch", h1_style))
    stock_intro = (
        "<b>1. What is a Stock?</b> A stock (equity) represents fractional ownership in a corporation (e.g., Apple, Reliance). Its price fluctuates based on supply and demand.<br/>"
        "<b>2. What is Bullish vs. Bearish?</b><br/>"
        "• <b>BULLISH (▲ UP):</b> Bulls attack by thrusting their horns UP. If market sentiment or technical momentum is bullish, the price is expected to rise. Investors want to BUY (Go Long).<br/>"
        "• <b>BEARISH (▼ DOWN):</b> Bears attack by swiping their paws DOWN. If signals are bearish, the price is expected to fall. Investors want to SELL, SHORT, or AVOID buying.<br/>"
        "<b>3. What is a Candlestick?</b> Instead of a simple line, candlesticks display 4 prices for each trading day (<b>OHLC</b>):<br/>"
        "• <b>Open (O):</b> Price when the market opened in the morning.<br/>"
        "• <b>High (H):</b> Highest price reached during the day.<br/>"
        "• <b>Low (L):</b> Lowest price reached during the day.<br/>"
        "• <b>Close (C):</b> Final price when the market closed at end of day.<br/>"
        "• <b>Green Candle:</b> Close &gt; Open (Stock went up during the day).<br/>"
        "• <b>Red Candle:</b> Close &lt; Open (Stock went down during the day).<br/>"
        "• <b>Wicks / Shadows:</b> The thin lines poking above and below the candle body showing the extreme High and Low prices.<br/>"
        "<b>4. What is Trading Volume?</b> The total number of shares bought and sold during that day (e.g., 42.32M shares). High volume confirms institutional backing; low volume means price moves are weak/unreliable.<br/>"
        "<b>5. What is Stop-Loss (SL) & Take-Profit (TP)?</b><br/>"
        "• <b>Stop-Loss:</b> A defensive price level where an investor automatically exits a trade to cut losses and protect capital (e.g., -4% from entry).<br/>"
        "• <b>Take-Profit:</b> A target price level where an investor locks in profits before the trend reverses (e.g., +6.5% from entry).<br/>"
        "• <b>Risk-to-Reward Ratio (R:R):</b> The ratio between potential loss and potential gain (e.g. 1 : 2.25 means risking ₹1 to make ₹2.25). Institutional funds require at least 1:1.5 to enter a trade."
    )
    story.append(Paragraph(stock_intro, body_style))
    story.append(Spacer(1, 6))

    # =========================================================================
    # PART 2: COMPLETE MASTER GLOSSARY OF ABBREVIATIONS
    # =========================================================================
    story.append(Paragraph("PART 2: Exhaustive Master Glossary of Abbreviations", h1_style))
    abbr_data = [
        [Paragraph("Abbreviation", table_header), Paragraph("Full Meaning", table_header), Paragraph("Plain-English Definition & Context in Ploutos+", table_header)],
        [Paragraph("LLM", table_cell_bold), Paragraph("Large Language Model", table_cell), Paragraph("Artificial intelligence model trained on vast text datasets (e.g., Qwen2.5-1.5B) to reason and generate predictions.", table_cell)],
        [Paragraph("MoE", table_cell_bold), Paragraph("Mixture of Experts", table_cell), Paragraph("Machine learning architecture where multiple specialized models ('experts') are coordinated by a gating 'router'.", table_cell)],
        [Paragraph("LoRA", table_cell_bold), Paragraph("Low-Rank Adaptation", table_cell), Paragraph("Efficient fine-tuning technique that freezes base model weights and trains small adapter rank matrices (r=16, α=32).", table_cell)],
        [Paragraph("PEFT", table_cell_bold), Paragraph("Parameter-Efficient Fine-Tuning", table_cell), Paragraph("Umbrella term for methods like LoRA that adapt massive LLMs without retraining all billions of parameters.", table_cell)],
        [Paragraph("MCC", table_cell_bold), Paragraph("Matthews Correlation Coefficient", table_cell), Paragraph("The golden standard metric for binary classification on imbalanced data (-1 = total disagreement, 0 = random guess, +1 = perfect).", table_cell)],
        [Paragraph("RSI", table_cell_bold), Paragraph("Relative Strength Index", table_cell), Paragraph("Momentum oscillator bounded between 0 and 100. Measures speed and change of price moves (Wilder 14-day formulation).", table_cell)],
        [Paragraph("OHLCV", table_cell_bold), Paragraph("Open, High, Low, Close, Volume", table_cell), Paragraph("The foundational 5 data points that describe trading action on any stock exchange.", table_cell)],
        [Paragraph("SMA", table_cell_bold), Paragraph("Simple Moving Average", table_cell), Paragraph("Average closing price over the last N days (e.g., SMA-5 is 5-day average; SMA-15 is 15-day average).", table_cell)],
        [Paragraph("R:R Ratio", table_cell_bold), Paragraph("Risk-to-Reward Ratio", table_cell), Paragraph("The mathematical ratio between downside risk distance and upside profit target distance.", table_cell)],
        [Paragraph("NSE / BSE", table_cell_bold), Paragraph("National / Bombay Stock Exchange", table_cell), Paragraph("India's primary equity exchanges. Tickers ending in .NS trade on NSE; .BO trade on BSE (denominated in ₹).", table_cell)],
        [Paragraph("P&L", table_cell_bold), Paragraph("Profit and Loss", table_cell), Paragraph("Realized financial gain or loss expressed in percentage or currency return upon settling a forward prediction.", table_cell)],
        [Paragraph("CUDA", table_cell_bold), Paragraph("Compute Unified Device Architecture", table_cell), Paragraph("NVIDIA's parallel computing platform allowing our RTX 4050 GPU to accelerate PyTorch neural matrix computations.", table_cell)],
        [Paragraph("FP16", table_cell_bold), Paragraph("16-Bit Half-Precision Floating Point", table_cell), Paragraph("Numerical precision format used during GPU training to halve VRAM memory consumption while doubling compute speed.", table_cell)]
    ]
    t_abbr = Table(abbr_data, colWidths=[70, 150, 296])
    t_abbr.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_abbr)
    story.append(Spacer(1, 8))

    # =========================================================================
    # PART 3: COMPLETE UI ELEMENT-BY-ELEMENT GUIDE (WHAT IS ON SCREEN)
    # =========================================================================
    story.append(Paragraph("PART 3: Element-by-Element UI Tour & Script for Professors", h1_style))
    ui_intro = (
        "When you open <b>http://127.0.0.1:8000</b> in your browser, every single widget, badge, and card on screen serves an institutional purpose. "
        "Here is what each element does and exactly what words to say when pointing at it:"
    )
    story.append(Paragraph(ui_intro, body_style))

    ui_elements = [
        (
            "1. Header Bar — Quick Tickers, Currency & LoRA Toggle",
            "• What it is: Top navigation bar containing quick stock buttons (`NVDA`, `AAPL`, `TSLA`, `MSFT`, `RELIANCE.NS`, `TCS.NS`), search input, and a `GPU LoRA` checkbox.<br/>"
            "• Currency Resolver: If you click `RELIANCE.NS` or `TCS.NS`, the system automatically detects the `.NS` suffix and formats everything in Indian Rupees (₹). For US stocks, it formats in Dollars ($).<br/>"
            "• GPU LoRA Checkbox: When checked, inferences run through our fine-tuned Qwen2.5-1.5B LoRA adapter on the local RTX 4050 GPU.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'Professor, here in the top navigation bar, we have multi-market support. When I click RELIANCE.NS, notice how the entire terminal dynamically switches all axes, tables, and guardrails to Indian Rupees (₹). The GPU LoRA toggle allows us to run inference through our locally fine-tuned adapter on our RTX 4050 GPU.'</i>"
        ),
        (
            "2. Top HUD Card 1 — Investment Capital Allocation Verdict",
            "• What it is: The leftmost KPI card displaying `DO NOT INVEST (AVOID)` in soft red or `STRONG BUY` in green, with subtext `0% Allocation (Exit Existing Longs / AVOID)`.<br/>"
            "• Why it is there: A raw prediction (UP/DOWN) is useless to an investor without knowing HOW MUCH capital to allocate. This card translates the ML confidence into a portfolio position sizing directive.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'Professor, this card is our Actionable Investment Verdict. Even if an asset is rising, if confidence is low or risk is asymmetric, the engine enforces strict capital preservation by recommending 0% allocation to protect client funds.'</i>"
        ),
        (
            "3. Top HUD Card 2 — Directional Forecast Signal & Vector",
            "• What it is: Shows `BEARISH ▼ DOWN` with a 🐻 Bear badge, or `BULLISH ▲ UP` with a 🐂 Bull badge, plus `65.0% Confidence • Next-Day Target Probability`.<br/>"
            "• Why it is there: Indicates whether the stock is forecasted to close higher or lower on the target trading day.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'This is our primary directional movement signal. It is derived by synthesizing technical momentum, retail sentiment, and visual candlestick wicks, outputting a calibrated probability percentage.'</i>"
        ),
        (
            "4. Top HUD Card 3 — Last Market Close & 5-Day Momentum",
            "• What it is: Displays the latest official market closing price (e.g. `₹1,307.90` or `$207.74`) and the 5-day trailing price return (e.g. `+9.34% 5-Day Momentum`).<br/>"
            "• Why it is there: Establishes the anchor price from which the next-day target return and stop-loss levels are computed.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'This card shows the benchmark close on day t-1. Notice that all our indicators are strictly lagged with zero lookahead leakage into target day t.'</i>"
        ),
        (
            "5. Top HUD Card 4 — Quantitative Risk Guardrails (Stop-Loss & Take-Profit)",
            "• What it is: Displays dynamic `STOP-LOSS` price (e.g. `₹1373.30 (-5%)`), `TAKE-PROFIT` target (e.g. `₹1203.27 (+8%)`), and `Risk-to-Reward Ratio: 1 : 1.60`.<br/>"
            "• How it is calculated: Uses the 5-day rolling return volatility ($\\sigma_{5d}$) to dynamically size the stop-loss distance. Higher volatility automatically widens stop-loss levels so normal market noise doesn't trigger premature exits.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'Professor, machine learning without risk management is reckless. Here, our engine calculates volatility-adjusted Stop-Loss and Take-Profit boundaries, ensuring the trade has a favorable risk-to-reward ratio.'</i>"
        ),
        (
            "6. Top HUD Card 5 — Dominant Expert & Empirical Proof Badge",
            "• What it is: Identifies which expert received majority weighting from the router (e.g. `Sentiment Expert`), a lead margin badge (`LEADS BY +5%` or `LEADS BY +18%`), and an empirical explanation.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'This card provides immediate interpretability. It proves that Ploutos+ is not a black box—it tells the portfolio manager exactly which data modality is leading today and by what mathematical spread.'</i>"
        ),
        (
            "7. Main Candlestick Chart (Left 65% Panel)",
            "• What it is: A full TradingView Lightweight Candlestick Chart with green/red candle bodies, upper/lower wick shadows, crosshair hover inspection (OHLC prices), and dual-panel bottom volume bars.<br/>"
            "• Badge: `Lagged Zero-Leakage (t < t+1)` proves no future data is displayed.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'On the left, we have an institutional TradingView candlestick interface. Each candle represents one trading day showing Open, High, Low, and Close prices. The bottom histogram shows trading volume in real time.'</i>"
        ),
        (
            "8. Adaptive Router Donut Ring (Right 35% Panel)",
            "• What it is: A dynamic ApexCharts donut ring displaying the three expert weights: Cyan = Technical, Amber = Sentiment, Purple = Human/Sector Proxy.<br/>"
            "• Center Text: Never displays the static word 'ROUTER'. Instead, it dynamically displays the dominant expert name and percentage (e.g. `SENTIMENT 41%` or `TECHNICAL 38%`).<br/>"
            "• Progress Bars: Horizontal spread breakdown bars showing the exact percentage allocation.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'Professor, this donut chart is the core methodological novelty of Ploutos+. In previous papers, expert weights were statically frozen at 33%. Our 16-D neural router dynamically reallocates weights based on real-time market conditions.'</i>"
        ),
        (
            "9. Multimodal Evidence Stack (3 Stacked Cards)",
            "• Card A (Technical Expert): Shows `RSI: 77.8`, `5d Vol: 0.86%`, `Vol Surge: +0.1x` with `BEARISH` badge.<br/>"
            "• Card B (Retail Social Sentiment Live Stream): Shows live StockTwits/news headlines with `Polarity: +0.00 NEUTRAL` and cue counters (`10 Tweets • 2 Bullish Cues • 0 Bearish Cues`).<br/>"
            "• Card C (Visual Candlestick Vision Expert): Shows detected chart geometry (e.g. `Exhaustion Shooting Star Reversal at Upper Bollinger Band` or `SMA-5 < SMA-15 Bearish Alignment`).<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'Here on the right, you can inspect the exact evidence from each modality. The portfolio manager can read the raw tweets, check the RSI indicator, and view the detected visual candlestick pattern.'</i>"
        ),
        (
            "10. Synthesized Multimodal Rationale",
            "• What it is: A qualitative paragraph generated by the fine-tuned LLM that weaves the technical, social, and visual evidence into a coherent rationale justifying the final directional call.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'This is the Rearview-Mirror Rationale. Our fine-tuned Qwen model explains in plain English why the evidence supports the directional prediction.'</i>"
        ),
        (
            "11. Bottom Tab 1 — Forward Tracking Ledger",
            "• What it is: A live connection to an ACID SQLite database (`results/live_tracker.db`) tracking every prediction forward in time.<br/>"
            "• Features: Summary KPI cards (Total Logged, Settled Calls, Forward Hit Rate %, Realized Return %) and a table with `HIT`, `MISS`, and `PENDING` settlement tags.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'Professor, backtesting alone can be curve-fitted. In Tab 1, we implemented an automated Forward Tracking Ledger. Every prediction is logged to a persistent SQLite database and settled the next market day to audit real-world forward performance.'</i>"
        ),
        (
            "12. Bottom Tab 2 — Historical Benchmark Audit (3,720 StockNet Records)",
            "• What it is: An interactive browser across all 3,720 held-out test records from the StockNet benchmark.<br/>"
            "• Features: Instant live ticker search/filter input (`Filter ticker...`) and clean columns displaying Ticker ($AAPL, $ABB), Date, Ground Truth (`▲ UP (1)` / `▼ DOWN (0)`), and Multi-Expert Prompts.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'In Tab 2, we expose our out-of-sample test split. Anyone can filter by ticker and verify our predictions against the actual historical ground-truth movements.'</i>"
        ),
        (
            "13. Bottom Tab 3 — Market-Regime Stress-Tests",
            "• What it is: Side-by-side grouped bar charts for Directional Accuracy % and MCC across 4 market regimes: Bull (956), Bear (629), Sideways (1,205), and High-Volatility (930).<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'Tab 3 proves our market-regime robustness. Notice that in volatile markets, Random Forest accuracy drops to 46% and its MCC becomes negative (-0.0103). Ploutos+ maintains positive MCC across all 4 regimes.'</i>"
        ),
        (
            "14. Bottom Tab 4 — Phase 13 Comprehensive Ablation Matrix",
            "• What it is: A full benchmark matrix comparing 9 model configurations (Majority, Sentiment alone, Fundamental alone, Static Ploutos, Visual alone, Technical alone, Random Forest, Router alone, Full Multimodal).<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'Tab 4 displays our ablation study. Our full multimodal system achieves 53.32% accuracy and +0.0596 MCC—a +27.1% relative gain over Random Forest.'</i>"
        ),
        (
            "15. Bottom Tab 5 — Quantitative Formulas & Live Proofs (The Math Tab)",
            "• What it is: Live interactive derivation cards showing Wilder RSI formula, 5-day volatility formula, volume surge ratio, decision scoring tree, and the **full 14-day worked table for RELIANCE.NS**.<br/>"
            "<b>• WHAT TO SAY TO PROFESSOR:</b> <i>'Professor, we eliminated all black-box handwaving. In Tab 5, we expose the exact step-by-step mathematical proofs. For RELIANCE.NS, you can follow every daily price change over 14 days and verify how the RSI of 77.8 is calculated line by line.'</i>"
        )
    ]

    for title, desc in ui_elements:
        story.append(Paragraph(title, h2_style))
        story.append(Paragraph(desc, body_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 6))

    # =========================================================================
    # PART 4: WORD-FOR-WORD PRESENTATION SCRIPT FOR THE 3 TEAMMATES
    # =========================================================================
    story.append(Paragraph("PART 4: Word-for-Word Presentation Speaking Script", h1_style))
    script_intro = (
        "Divide the 15-minute presentation cleanly into three 5-minute segments so each team member speaks with total authority:"
    )
    story.append(Paragraph(script_intro, body_style))

    speakers = [
        (
            "Speaker 1: Rishikesh (Roll: 24BDS030) — The Foundation & LoRA Fine-Tuning",
            "<i>'Good morning, Professor. Today, our team is presenting Ploutos+, an interpretable multimodal stock movement prediction system powered by financial Large Language Models.<br/>"
            "The fundamental problem in existing financial NLP is that models rely on static heuristics—fixing weights at 33% regardless of whether there is breaking news or market calm. Furthermore, many models suffer from data leakage.<br/>"
            "To solve this, my responsibility was the data engineering and fine-tuning pipeline. We acquired 87 stocks from the StockNet benchmark spanning 26,623 records. We enforced a strict chronological split without lookahead bias: our training data spans 2014 to mid-2015, and our held-out test set spans October 2015 to January 2016.<br/>"
            "We fine-tuned Qwen2.5-1.5B using PEFT LoRA on attention projection layers using our local RTX 4050 GPU. Training took just 6.00 minutes, and the training loss converged smoothly from 7.03 down to 0.2737. Now, my teammate Vineel will explain our Adaptive Expert Router and market-regime stress testing.'</i>"
        ),
        (
            "Speaker 2: Vineel (Roll: 24BDS010) — The Adaptive Router & Visual Vision Expert",
            "<i>'Thank you, Rishikesh. My core contribution was the Adaptive Context-Gated Mixture-of-Experts Router and the Visual Financial Expert.<br/>"
            "Instead of static weights, our router extracts a 16-dimensional lagged context vector capturing price momentum, oscillator bands, volatility, and sentiment density. Through a temperature-scaled softmax gating network, it dynamically allocates weights among technical, sentiment, and fundamental experts.<br/>"
            "Additionally, we built the Visual Financial Expert, which reconstructs 15-day OHLC candlestick charts to detect spatial patterns like pin-bar hammer reversals, shooting-star rejections, and moving-average crossovers with zero temporal leakage.<br/>"
            "In Phase 10, we stress-tested Ploutos+ across four economic regimes: Bull, Bear, Sideways, and Volatile. As you can see in Tab 3 of our dashboard, Random Forest breaks down completely in volatile markets, dropping to an MCC of -0.0103. In contrast, Ploutos+ maintains positive MCC across all four regimes. Now, Arjun will demonstrate our live institutional terminal and mathematical proofs.'</i>"
        ),
        (
            "Speaker 3: Arjun (Roll: 24BDS039) — Live Terminal, Math Proofs & Risk Allocation",
            "<i>'Thank you, Vineel. I was responsible for our live ingestion pipeline, our FastAPI TradingView terminal, our mathematical proofs engine, and our actionable risk guardrails.<br/>"
            "If you look at our live terminal at port 8000, we built a real-time system that ingests live quotes from yfinance and streaming retail sentiment from StockTwits. We implemented dynamic multi-currency resolution: Indian stocks like RELIANCE.NS automatically render in Rupees (₹), while US stocks render in Dollars ($).<br/>"
            "Crucially, we do not just predict UP or DOWN; we generate actionable risk guardrails. Card 4 calculates dynamic Stop-Loss and Take-Profit target boundaries based on rolling volatility, while Card 1 outputs a capital allocation verdict like DO NOT INVEST or STRONG BUY.<br/>"
            "Finally, in Tab 5, we expose verifiable mathematical proofs. For RELIANCE.NS, we provide a complete 14-day worked table showing how Wilder RSI-14 is derived step-by-step from raw closes to 77.8. In Phase 13, our full multimodal system achieved 53.32% accuracy and an MCC of +0.0596—a +27.1% relative gain over Random Forest. We are now open to your questions.'</i>"
        )
    ]

    for speaker_title, script in speakers:
        story.append(Paragraph(speaker_title, h2_style))
        t_sc = Table([[Paragraph(script, script_box_style)]], colWidths=[516])
        t_sc.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#ECFDF5")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#10B981")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_sc)
        story.append(Spacer(1, 4))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[Beginner Guide PDF Compiled] Successfully generated at: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_guide_pdf()

