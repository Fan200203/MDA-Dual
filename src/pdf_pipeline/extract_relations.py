from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from docx import Document
from pypdf import PdfReader
from tqdm import tqdm

from src.pdf_pipeline.prompt_schema import SYSTEM_PROMPT


LOGGER = logging.getLogger("pdf_llm_extraction")
OUTPUT_COLUMNS = ["pmid", "microbe", "disease", "effect", "evidence", "SCD", "source_file"]


def extract_pmid(filename: str) -> str:
    match = re.search(r"PMID[_ -]?(\d+)", filename, flags=re.IGNORECASE)
    return match.group(1) if match else f"UNKNOWN_{Path(filename).stem}"


def read_document(path: Path, max_pdf_pages: int = 10) -> tuple[str, str]:
    pmid = extract_pmid(path.name)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join((reader.pages[i].extract_text() or "") for i in range(min(max_pdf_pages, len(reader.pages))))
    elif suffix == ".docx":
        text = "\n".join(paragraph.text for paragraph in Document(str(path)).paragraphs)
    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    return text, pmid


def call_deepseek(
    text: str,
    api_key: str,
    base_url: str,
    model: str,
    max_input_characters: int,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
    retry_delay: int,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"文献内容：\n{text[:max_input_characters]}"},
        ],
        "max_tokens": max_output_tokens,
        "temperature": 0.3,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            LOGGER.exception("API attempt %d/%d failed", attempt, max_retries)
            if attempt < max_retries:
                time.sleep(retry_delay)
    raise RuntimeError("DeepSeek API failed after all retries")


def parse_json_response(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text or "", flags=re.IGNORECASE)
    candidate = (match.group(1) if match else text or "{}").strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            return json.loads(candidate[start : end + 1])
        raise


def relation_rows(payload: dict, pmid: str, source_file: str) -> list[dict[str, str]]:
    scd = str(payload.get("cytokine_signaling_disease", "") or "").strip()
    rows = []
    for item in payload.get("microbe_disease_relationships", []) or []:
        effect = str(item.get("effect", "")).strip().lower()
        if effect not in {"increase", "decrease"}:
            continue
        microbe = str(item.get("microbe", "")).strip()
        disease = str(item.get("disease", "")).strip()
        if not microbe or not disease:
            continue
        rows.append(
            {
                "pmid": pmid,
                "microbe": microbe,
                "disease": disease,
                "effect": effect,
                "evidence": str(item.get("evidence", "") or "").strip(),
                "SCD": scd,
                "source_file": source_file,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Read PDF/DOCX/TXT files and extract directed microbe-disease relations with the original few-shot/CoT prompt.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pdf_pipeline/extracted_relations.csv"))
    parser.add_argument("--failures", type=Path, default=Path("artifacts/pdf_pipeline/failed_files.txt"))
    parser.add_argument("--api-base", default=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--max-pdf-pages", type=int, default=10)
    parser.add_argument("--max-input-characters", type=int, default=8000)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=5)
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY in the environment; never commit a real key.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {".pdf", ".docx", ".txt"})
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for path in tqdm(files, desc="Literature extraction"):
        try:
            text, pmid = read_document(path, args.max_pdf_pages)
            if not text.strip():
                raise ValueError("No extractable text")
            response = call_deepseek(
                text,
                api_key,
                args.api_base,
                args.model,
                args.max_input_characters,
                args.max_output_tokens,
                args.timeout,
                args.max_retries,
                args.retry_delay,
            )
            rows.extend(relation_rows(parse_json_response(response), pmid, path.name))
        except Exception as exc:
            LOGGER.exception("Failed: %s", path)
            failures.append(f"{path.name}\t{type(exc).__name__}: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(args.output, index=False, encoding="utf-8-sig")
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    args.failures.write_text("\n".join(failures), encoding="utf-8")
    print({"documents": len(files), "relations": len(rows), "failures": len(failures), "output": str(args.output)})


if __name__ == "__main__":
    main()
