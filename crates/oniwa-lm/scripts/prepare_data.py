"""中島敦『山月記』のデータ取得・ルビクリーニング・トークナイズ・系譜台帳記録スクリプト"""

import hashlib
import json
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

AOZORA_URL = "https://www.aozora.gr.jp/cards/000119/files/624_ruby_5668.zip"
SOURCE_NAME = "青空文庫: 中島敦『山月記』"
LICENSE = "Public Domain (著作権保護期間満了)"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def clean_aozora_text(text: str) -> str:
    """青空文庫のルビ・注記・ヘッダー・フッターを除去して本文のみを抽出"""
    # ヘッダー除去 (------------------------------------------------------- の間)
    lines = text.splitlines()
    body_lines = []
    in_header = True
    dash_count = 0

    for line in lines:
        if "-------------------------------------------------------" in line:
            dash_count += 1
            if dash_count == 2:
                in_header = False
            continue
        if in_header:
            continue
        # 底本以降のフッターは除外
        if line.startswith("底本：") or line.startswith("［＃"):
            if "底本：" in line:
                break
        body_lines.append(line)

    body = "\n".join(body_lines)

    # 1. ルビの除去: 《...》
    body = re.sub(r"《.*?》", "", body)
    # 2. 入力者注記の除去: ［＃...］
    body = re.sub(r"［＃.*?］", "", body)
    # 3. 連続する空行の正規化
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def main():
    print(f"[*] Downloading: {AOZORA_URL} ...")
    zip_path = DATA_DIR / "sangetsuki.zip"

    req = urllib.request.Request(
        AOZORA_URL,
        headers={"User-Agent": "niwa-lm-clean-ai-project/0.1.0"},
    )
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
        f.write(resp.read())

    # ZIP展開
    print("[*] Extracting and decoding Shift_JIS ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        txt_filename = [name for name in z.namelist() if name.endswith(".txt")][0]
        raw_bytes = z.read(txt_filename)

    raw_text = raw_bytes.decode("cp932", errors="replace")
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # クリーニング
    clean_text = clean_aozora_text(raw_text)
    clean_txt_path = DATA_DIR / "sangetsuki_clean.txt"
    clean_txt_path.write_text(clean_text, encoding="utf-8")
    print(f"[*] Clean text saved to {clean_txt_path} ({len(clean_text)} chars)")
    print(f"    Preview: {clean_text[:60]}...")

    # 文字単位トークナイザーの構築
    chars = sorted(list(set(clean_text)))
    vocab = {ch: i for i, ch in enumerate(chars)}
    inv_vocab = {i: ch for i, ch in enumerate(chars)}

    vocab_path = DATA_DIR / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {"vocab": vocab, "inv_vocab": inv_vocab, "size": len(vocab)},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[*] Vocab size: {len(vocab)} unique characters")

    # トークン列を uint16 バイナリとして保存
    token_ids = [vocab[ch] for ch in clean_text]
    bin_path = DATA_DIR / "tokens.bin"

    import struct

    with open(bin_path, "wb") as f:
        for tid in token_ids:
            f.write(struct.pack("<H", tid))  # Little-endian uint16

    tokenized_bytes = bin_path.read_bytes()
    tokenized_sha256 = hashlib.sha256(tokenized_bytes).hexdigest()

    print(
        f"[*] Saved {len(token_ids)} tokens to {bin_path} (SHA-256: {tokenized_sha256[:16]}...)"
    )

    # 系譜台帳 (ledger_index.jsonl) への記録
    ledger_path = LOGS_DIR / "ledger_index.jsonl"
    ingestion_event = {
        "event_type": "DataIngestion",
        "payload": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_name": SOURCE_NAME,
            "source_url_or_path": AOZORA_URL,
            "license": LICENSE,
            "raw_data_sha256": raw_sha256,
            "raw_data_bytes": len(raw_bytes),
            "tokenized_sha256": tokenized_sha256,
            "num_tokens": len(token_ids),
            "vocab_size": len(vocab),
            "tokenizer_type": "Character-level (UTF-8)",
        },
    }

    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ingestion_event, ensure_ascii=False) + "\n")

    print(f"[+] Recorded DataIngestion event to {ledger_path}")


if __name__ == "__main__":
    main()
