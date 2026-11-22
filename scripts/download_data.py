"""下载公开数据集并校验 SHA-256。

数据源：
- FGRC-SCD（MIT）：电信诈骗风险短信，HuggingFace Abooooo/FGRC-SCD
- mudou_spam（许可证未注明）：中文垃圾短信，GitHub shaonianruntu/Spam-Message-Classification
- FBS_SMS_Dataset（需引用 CCS'20 论文）：伪基站垃圾短信，GitHub Cypher-Z/FBS_SMS_Dataset

用法： python scripts/download_data.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.antifraud.config import RAW_DIR, ensure_dirs

FGRC_URL = "https://hf-mirror.com/datasets/Abooooo/FGRC-SCD/resolve/main/FGRC-SCD-sms.zip"
FGRC_FALLBACK = "https://huggingface.co/datasets/Abooooo/FGRC-SCD/resolve/main/FGRC-SCD-sms.zip"

MUDOU_BASE = "https://raw.githubusercontent.com/shaonianruntu/Spam-Message-Classification/master"

FBS_BASE = "https://raw.githubusercontent.com/Cypher-Z/FBS_SMS_Dataset/master"
FBS_FILES = [
    "AD:Loan", "AD:Network_service", "AD:Other", "AD:Real_estate", "AD:Retail",
    "FR:Financial", "FR:Other", "FR:Phishing(Bank)", "FR:Phishing(Other)",
    "IL:Escort_service", "IL:Fake_ID_and_invoice", "IL:Gambling",
    "IL:Political_propaganda", "Other",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: int = 600) -> bool:
    if dest.exists():
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "campus-antifraud-agent"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
                open(dest, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
        return True
    except Exception as e:
        print(f"  下载失败 {url}: {e}")
        return False


def main() -> None:
    ensure_dirs()
    manifest: dict = {}

    # 1) FGRC-SCD
    fgrc_dir = RAW_DIR / "fgrc_scd"
    fgrc_dir.mkdir(parents=True, exist_ok=True)
    zip_path = fgrc_dir / "FGRC-SCD-sms.zip"
    print("[1/3] 下载 FGRC-SCD 短信数据集 ...")
    if not _download(FGRC_URL, zip_path):
        _download(FGRC_FALLBACK, zip_path)
    if zip_path.exists() and not (fgrc_dir / "message").exists():
        with ZipFile(zip_path) as z:
            z.extractall(fgrc_dir)
    manifest["fgrc_scd_sms_zip_sha256"] = _sha256(zip_path)

    # 2) mudou
    mudou_dir = RAW_DIR / "mudou"
    mudou_dir.mkdir(parents=True, exist_ok=True)
    print("[2/3] 下载 mudou 中文垃圾短信 ...")
    for name in ["mudou_spam.train", "mudou_spam.test"]:
        _download(f"{MUDOU_BASE}/{name}", mudou_dir / name)
        manifest[f"mudou_{name}_sha256"] = _sha256(mudou_dir / name)

    # 3) FBS
    fbs_dir = RAW_DIR / "fbs"
    fbs_dir.mkdir(parents=True, exist_ok=True)
    print("[3/3] 下载 FBS 伪基站短信（14 类）...")
    from urllib.parse import quote
    for name in FBS_FILES:
        url = f"{FBS_BASE}/{quote(name, safe='')}"
        _download(url, fbs_dir / name.replace(":", "_").replace("(", "_").replace(")", ""))
        manifest[f"fbs_{name}"] = _sha256(fbs_dir / name.replace(":", "_").replace("(", "_").replace(")", ""))

    manifest_path = RAW_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("完成，清单已写入", manifest_path)


if __name__ == "__main__":
    main()
