"""用系统 Edge（Playwright 驱动）对 Streamlit 应用截图，用于报告演示。

前置：Streamlit 应用需已在本机运行（默认 http://localhost:8501）。

用法： python scripts/screenshot_ui.py [输出目录]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.antifraud.config import REPORT_DIR

URL = "http://localhost:8501"

SCAM_TEXT = "我是公安局民警，你名下银行卡涉嫌洗钱，请立即把资金转到安全账户验证，否则冻结账户。"


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1500})

        # 1) 主页面
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=校园反诈助手", timeout=30000)
        time.sleep(2)
        page.screenshot(path=str(out_dir / "ui_main.png"), full_page=True)

        # 2) 输入诈骗短信并检测
        page.locator("textarea").first.fill(SCAM_TEXT)
        page.get_by_role("button", name="开始检测").click()
        page.wait_for_selector("text=综合风险分", timeout=30000)
        time.sleep(1)
        page.screenshot(path=str(out_dir / "ui_scam_result.png"), full_page=True)

        browser.close()
    print("截图完成：", out_dir)


if __name__ == "__main__":
    main()
