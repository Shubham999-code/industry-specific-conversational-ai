#!/usr/bin/env python3
"""
Data Preparation, Sanitization, and Curation Pipeline for Industry-Specific Conversational AI
Target Domain: Technology & IT Service Management (ITSM) / Cloud SRE Incident Response

This script preprocesses raw technical postmortems, incident logs, and multi-turn ticket threads
into clean, instruction-tuned conversational datasets ready for Hugging Face LoRA fine-tuning.
"""

import json
import re
import os
import argparse
from typing import List, Dict, Any

# Regular expressions for redaction and compliance
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
AWS_KEY_REGEX = re.compile(r"\b(AKIA[0-9A-Z]{16})\b")
SECRET_KEY_REGEX = re.compile(r"(?i)(api[_-]?key|secret|password|bearer|auth|token)[\s:=]+['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")

# RFC 5737 documentation IP ranges
REPLACEMENT_IP = "192.0.2.10"
REDACTED_SECRET = "<SECRET_REDACTED>"

def sanitize_text(text: str) -> str:
    """Sanitizes PII, secret keys, production IPs, and sensitive credentials."""
    if not isinstance(text, str):
        return text
    text = AWS_KEY_REGEX.sub("AKIAIOSFODNN7EXAMPLE", text)
    text = SECRET_KEY_REGEX.sub(r"\1: <SECRET_REDACTED>", text)
    text = EMAIL_REGEX.sub("devops-engineer@enterprise.internal", text)
    text = IP_REGEX.sub(REPLACEMENT_IP, text)
    return text

def sanitize_conversation(record: Dict[str, Any]) -> Dict[str, Any]:
    """Applies sanitization across all message turns in a dialogue record."""
    sanitized_messages = []
    for msg in record.get("messages", []):
        sanitized_messages.append({
            "role": msg["role"],
            "content": sanitize_text(msg["content"])
        })
    record["messages"] = sanitized_messages
    return record

def format_llama3_chat(messages: List[Dict[str, str]]) -> str:
    """Formats messages into Llama-3 instruction chat template."""
    formatted = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    return formatted

def compute_dataset_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates dataset health metrics: token lengths, turns, category counts."""
    total_turns = 0
    domains = {}
    severities = {}
    char_lengths = []

    for rec in records:
        turns = len(rec.get("messages", []))
        total_turns += turns
        domain = rec.get("domain", "General IT")
        severity = rec.get("severity", "Unclassified")
        domains[domain] = domains.get(domain, 0) + 1
        severities[severity] = severities.get(severity, 0) + 1
        for msg in rec.get("messages", []):
            char_lengths.append(len(msg.get("content", "")))

    avg_chars = sum(char_lengths) / max(len(char_lengths), 1)
    return {
        "total_records": len(records),
        "total_dialogue_turns": total_turns,
        "average_turns_per_dialogue": round(total_turns / max(len(records), 1), 2),
        "average_message_chars": round(avg_chars, 2),
        "estimated_average_tokens": round(avg_chars / 4.0, 1),
        "domain_distribution": domains,
        "severity_distribution": severities
    }

def main():
    parser = argparse.ArgumentParser(description="Preprocess and curate ITSM conversational datasets")
    parser.add_argument("--input_file", type=str, default="dataset/itsm_devops_curated_dataset.jsonl", help="Raw dataset file")
    parser.add_argument("--output_dir", type=str, default="dataset/processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    if os.path.exists(args.input_file):
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    # Sanitize records
    sanitized_records = [sanitize_conversation(r) for r in records]

    # Compute statistics
    stats = compute_dataset_statistics(sanitized_records)
    print("=== Dataset Curation & Health Summary ===")
    print(json.dumps(stats, indent=2))

    # Save processed outputs
    processed_jsonl = os.path.join(args.output_dir, "train_curated.jsonl")
    with open(processed_jsonl, "w", encoding="utf-8") as f:
        for r in sanitized_records:
            f.write(json.dumps(r) + "\n")

    stats_file = os.path.join(args.output_dir, "dataset_curation_report.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Curated dataset successfully saved to: {processed_jsonl}")

if __name__ == "__main__":
    main()
