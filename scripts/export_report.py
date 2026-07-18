#!/usr/bin/env python3
"""Build submission REPORT.docx and REPORT.pdf with embedded screenshots."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "REPORT.md"
OUT_MD = ROOT / "REPORT_EXPORT.md"
OUT_DOCX = ROOT / "REPORT.docx"
OUT_PDF = ROOT / "REPORT.pdf"
OUT_HTML = ROOT / "REPORT.html"

# Replace plain screenshot path mentions with markdown images
IMAGE_BLOCKS = {
    r"\*\*EDA screenshots:\*\*.*": """### EDA figures

![Class balance](screenshots/class_balance.png)

![Correlation heatmap](screenshots/correlation_heatmap.png)

![Numeric histograms](screenshots/numeric_histograms.png)
""",
    r"\*\*MLflow screenshot:\*\*.*": """### MLflow experiment tracking

![MLflow UI — heart-disease-uci runs](screenshots/02_mlflow_ui.png)
""",
    r"\*\*CI screenshot:\*\*.*": """### CI/CD evidence

![GitHub Actions green run](screenshots/04_github_actions.png)
""",
    r"\*\*Docker screenshots:\*\*.*": """### Docker serving evidence

![Docker /health and /predict](screenshots/05_docker_predict.png)

![FastAPI Swagger UI](screenshots/08_swagger_ui.png)
""",
    r"\*\*Kubernetes screenshots:\*\*.*": """### Kubernetes deployment evidence

![Kubernetes pods and services](screenshots/06_k8s_pods.png)
""",
    r"\*\*Monitoring screenshots:\*\*.*": """### Monitoring evidence

![Prometheus targets UI](screenshots/07_prometheus_targets.png)
""",
}


def build_export_markdown() -> Path:
    text = SRC.read_text(encoding="utf-8")
    # Drop leftover placeholder note
    text = text.replace(
        "*(After training, paste your leaderboard numbers and best model name here.)*\n\n",
        "",
    )
    # Insert model comparison chart near results section
    if "model_comparison_roc_auc.png" not in text:
        text = text.replace(
            "## 13. Results Summary",
            """## 13. Results Summary

![Model comparison ROC-AUC](screenshots/model_comparison_roc_auc.png)
""",
        )
    for pattern, replacement in IMAGE_BLOCKS.items():
        text = re.sub(pattern, replacement, text)
    OUT_MD.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    return OUT_MD


def export_docx(md: Path) -> None:
    subprocess.run(
        [
            "pandoc",
            str(md),
            "-o",
            str(OUT_DOCX),
            f"--resource-path={ROOT}",
            "--from",
            "markdown",
            "--to",
            "docx",
            "--toc",
            "--toc-depth=2",
            "-V",
            "geometry:margin=1in",
        ],
        check=True,
        cwd=ROOT,
    )
    print(f"Wrote {OUT_DOCX} ({OUT_DOCX.stat().st_size // 1024} KB)")


def export_html(md: Path) -> Path:
    css = """
    @page { size: A4; margin: 14mm 14mm 14mm 14mm; }
    body { font-family: Georgia, 'Times New Roman', serif; max-width: 800px; margin: 12px auto; line-height: 1.32; color: #111; font-size: 11pt; }
    h1 { font-size: 18pt; margin: 0.4em 0; font-family: Arial, Helvetica, sans-serif; color: #1a2a3a; }
    h2 { font-size: 14pt; margin: 0.7em 0 0.3em; font-family: Arial, Helvetica, sans-serif; color: #1a2a3a; page-break-after: avoid; }
    h3 { font-size: 12pt; margin: 0.5em 0 0.25em; font-family: Arial, Helvetica, sans-serif; color: #1a2a3a; page-break-after: avoid; }
    p, li { margin: 0.25em 0; }
    img { max-width: 80%; max-height: 210px; width: auto; height: auto; object-fit: contain; border: 1px solid #ddd; margin: 4px auto 8px; display: block; page-break-inside: avoid; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 10pt; }
    th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; }
    code, pre { font-family: Menlo, Consolas, monospace; font-size: 0.85em; }
    pre { background: #f6f8fa; padding: 8px; overflow-x: auto; margin: 6px 0; }
    #TOC { font-size: 10pt; }
    """
    css_path = ROOT / "artifacts" / "report.css"
    css_path.parent.mkdir(parents=True, exist_ok=True)
    css_path.write_text(css, encoding="utf-8")
    subprocess.run(
        [
            "pandoc",
            str(md),
            "-o",
            str(OUT_HTML),
            f"--resource-path={ROOT}",
            "--standalone",
            "--toc",
            "--toc-depth=2",
            f"--css={css_path}",
            "--metadata",
            "title=MLOps Assignment 01 – Heart Disease Risk Prediction",
        ],
        check=True,
        cwd=ROOT,
    )
    print(f"Wrote {OUT_HTML}")
    return OUT_HTML


def export_pdf_via_playwright(html: Path) -> None:
    from playwright.sync_api import sync_playwright

    url = html.resolve().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=120000)
        page.pdf(
            path=str(OUT_PDF),
            format="A4",
            print_background=True,
            margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
        )
        browser.close()
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size // 1024} KB)")


def page_count_estimate(pdf: Path) -> None:
    # Rough estimate from PDF size / content; try pypdf if available
    try:
        from pypdf import PdfReader

        n = len(PdfReader(str(pdf)).pages)
        print(f"PDF page count: {n}")
    except Exception:
        print("Install pypdf for exact page count (optional).")


def main() -> None:
    md = build_export_markdown()
    export_docx(md)
    html = export_html(md)
    export_pdf_via_playwright(html)
    try:
        subprocess.run(
            [str(ROOT / ".venv" / "bin" / "pip"), "install", "pypdf", "-q"],
            check=False,
        )
    except Exception:
        pass
    page_count_estimate(OUT_PDF)
    print("\nSubmission files ready:")
    print(f"  - {OUT_DOCX}")
    print(f"  - {OUT_PDF}")


if __name__ == "__main__":
    main()
