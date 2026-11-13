"""Fail if tracked files look like they contain credentials. A pragmatic regex scan, not a replacement for a
dedicated scanner; it exists so CI has an executable check.

    python scripts/check_secrets.py            # scans `git ls-files`
"""
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "AWS access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "Slack/Discord webhook": re.compile(r"https://(?:hooks\.slack\.com/services|discord(?:app)?\.com/api/webhooks)/[A-Za-z0-9/_-]{10,}"),
    "assigned secret": re.compile(r"""(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['"][A-Za-z0-9/+_\-]{20,}['"]"""),
}
FORBIDDEN_FILES = (".env",)          # tracked env files are never allowed (.env.example is fine)
SKIP_SUFFIX = (".parquet", ".png", ".jpg", ".joblib", ".pkl", ".sqlite", ".zip", ".ipynb")


def scan_text(text: str, path: str = "<text>") -> list[str]:
    out = []
    for name, pat in PATTERNS.items():
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            out.append(f"{path}:{line}: looks like a {name}")
    return out


def main() -> int:
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    findings: list[str] = []
    for f in filter(None, files):
        if Path(f).name in FORBIDDEN_FILES:
            findings.append(f"{f}: environment file must not be tracked")
        if f.endswith(SKIP_SUFFIX) or not Path(f).is_file():
            continue
        try:
            findings += scan_text(Path(f).read_text(errors="ignore"), f)
        except OSError:
            continue
    for x in findings:
        print(x)
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
