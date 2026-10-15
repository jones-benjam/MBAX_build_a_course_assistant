"""Capture UI screenshots with Playwright.  Usage:
    python app.py &                              # app running on :7860
    python scripts/screenshot_ui.py docs/screenshots light
    python scripts/screenshot_ui.py docs/screenshots dark
Set CHROME_PATH to use a preinstalled Chromium instead of `playwright install`."""
import os
import sys
from playwright.sync_api import sync_playwright

out = sys.argv[1]
theme = sys.argv[2] if len(sys.argv) > 2 else "light"
url = f"http://127.0.0.1:7860/?__theme={theme}"

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None, args=["--no-proxy-server"])
    page = b.new_page(viewport={"width": 1400, "height": 1000}, color_scheme=theme)
    page.goto(url, wait_until="domcontentloaded"); page.get_by_role("tab", name="Ask").wait_for(timeout=30000)
    page.wait_for_timeout(1500)
    page.screenshot(path=f"{out}/materials-{theme}.png", full_page=True)

    # Ask tab
    page.get_by_role("tab", name="Ask").click()
    page.wait_for_timeout(500)
    page.get_by_label("Your question").fill("Find the meme about Vibe Coding on Prod in the Week 2 slides and explain what it shows")
    page.get_by_role("button", name="Ask", exact=True).click()
    page.wait_for_timeout(4000)
    page.screenshot(path=f"{out}/ask-{theme}.png", full_page=True)

    # Quiz tab
    page.get_by_role("tab", name="Quiz").click()
    page.wait_for_timeout(500)
    page.get_by_label("Topic filter (optional)").fill("retrieval augmented generation")
    page.get_by_role("button", name="Generate quiz").click()
    page.wait_for_timeout(4000)
    # pick one option on every visible question
    labels = page.locator("label:has(input[type=radio])")
    seen = set()
    for i in range(labels.count()):
        el = labels.nth(i)
        if not el.is_visible():
            continue
        name = el.locator("input").get_attribute("name")
        if name in seen:
            continue
        seen.add(name)
        el.click()
    page.get_by_role("button", name="Submit answers").click()
    page.wait_for_timeout(2500)
    page.screenshot(path=f"{out}/quiz-{theme}.png", full_page=True)
    b.close()
print("done", theme)
