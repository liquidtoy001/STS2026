"""Check docs/site translations against the data-i18n keys used in the pages.

Missing keys are allowed (the page falls back to English) and are listed for
translators. Duplicate keys in the pages and dictionary keys that no page uses
are errors, because they usually mean a page was edited without its translations.

Usage: python tools/check_site_i18n.py [--missing]
"""
import re
import sys
from collections import Counter
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "docs" / "site"
KEY_IN_PAGE = re.compile(r'data-i18n="([^"]+)"')
KEY_IN_DICT = re.compile(r'^\s*"((?:ov|v0|v1|imp)\.[\w-]+)"\s*:', re.M)


def main():
    show_missing = "--missing" in sys.argv
    used = Counter()
    for page in sorted(SITE.glob("*.html")):
        used.update(KEY_IN_PAGE.findall(page.read_text(encoding="utf-8")))
    errors = [f"duplicate page key {k} ({n}x)" for k, n in used.items() if n > 1]

    for path in sorted((SITE / "assets" / "i18n").glob("*.js")):
        keys = set(KEY_IN_DICT.findall(path.read_text(encoding="utf-8")))
        stale = sorted(keys - set(used))
        missing = sorted(set(used) - keys)
        errors += [f"{path.name}: key {k} is not used by any page" for k in stale]
        print(f"{path.stem}: {len(keys)} translated, {len(missing)} fall back to English")
        if show_missing and missing:
            print("  " + " ".join(missing))

    for e in errors:
        print("ERROR:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
