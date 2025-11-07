import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path

# ==================== CONFIG ====================
# Folder containing raw WhatsApp exports (.txt as downloaded from the app)
RAW_TXT_FOLDER = "data/unformatted_text"   # change to your actual folder if needed

# Folder for normalized text format: "[dd.mm.yy, HH:MM] Name: Message"
NORMALIZED_FOLDER = "data/raw_data"

# Folder for final JSON files in ShareGPT-style format
OUTPUT_JSON_FOLDER = "data/preprocessed"

# Max messages per JSON chunk (to avoid huge conversations)
MAX_MESSAGES_PER_FILE = 40
MIN_MESSAGES_PER_FILE = 3
# =================================================


# WhatsApp export format:
# "12/03/2024, 17:42 - Name: Message"
LINE_EXPORT_RE = re.compile(
    r'^(\d{1,2})/(\d{1,2})/(\d{4}), (\d{1,2}):(\d{2}) - ([^:]+): (.*)$'
)

# Normalized format:
# "[12.03.24, 17:42] Name: Message"
LINE_NORMALIZED_RE = re.compile(
    r'^\[(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}(?::\d{2})?)\] (.+?): (.+)$'
)


def parse_whatsapp_export(path):
    """
    Parse a WhatsApp .txt export into a list of messages:
    [{"dt": datetime, "name": str, "text": str}, ...]
    Handles multiline messages by attaching them to the previous line.
    """
    messages = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")

            m = LINE_EXPORT_RE.match(raw)
            if m:
                d, mo, y, hh, mm, name, text = m.groups()
                dt = datetime(int(y), int(mo), int(d), int(hh), int(mm))
                messages.append({"dt": dt, "name": name.strip(), "text": text.strip()})
            else:
                # Continuation lines: append to previous message if non-empty
                if messages and raw.strip():
                    messages[-1]["text"] = (
                        messages[-1]["text"] + "\n" + raw.strip()
                    ).strip()
                # Otherwise ignore system/empty lines we don't recognize

    messages.sort(key=lambda x: x["dt"])
    return messages


def format_normalized_line(msg):
    """
    Convert internal msg dict to normalized string:
    "[dd.mm.yy, HH:MM] Name: Message"
    """
    date_str = msg["dt"].strftime("%d.%m.%y")
    time_str = msg["dt"].strftime("%H:%M")
    return f"[{date_str}, {time_str}] {msg['name']}: {msg['text']}"


def normalize_all_chats(raw_folder, out_folder):
    """
    Read all .txt exports from `raw_folder`,
    parse them, and write normalized text chats to `out_folder`.
    """
    os.makedirs(out_folder, exist_ok=True)

    txt_files = sorted(Path(raw_folder).glob("*.txt"))
    if not txt_files:
        print(f"[WARN] No .txt files found in {raw_folder}")
        return

    for path in txt_files:
        msgs = parse_whatsapp_export(path)
        if not msgs:
            print(f"[WARN] {path.name}: no valid messages parsed, skipping.")
            continue

        out_path = Path(out_folder) / path.name
        with open(out_path, "w", encoding="utf-8") as f:
            for m in msgs:
                f.write(format_normalized_line(m) + "\n")

        print(f"[OK] Normalized: {path.name} -> {out_path}")


def txt_to_json_from_normalized(txt_file_path, json_base_path, whatsapp_name):
    """
    Convert a normalized chat file:
        [dd.mm.yy, HH:MM] Name: Message
    into one or more JSON files in ShareGPT-like format:
        {"conversations": [{"from": "gpt"/"human", "value": "..."}]}
    """
    conversations = []

    with open(txt_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            m = LINE_NORMALIZED_RE.match(line)
            if not m:
                continue

            sender = m.group(3).strip()
            content = m.group(4).strip()

            # Skip missed call artifacts
            if "Missed video call" in content or "Missed voice call" in content:
                continue

            # Tag messages: from you -> "gpt", others -> "human"
            from_field = "gpt" if whatsapp_name in sender else "human"

            conversations.append({
                "from": from_field,
                "value": content
            })

    # Not enough content: skip
    if len(conversations) < MIN_MESSAGES_PER_FILE:
        return

    # If very long, split into multiple JSONs
    if len(conversations) > MAX_MESSAGES_PER_FILE:
        for i in range(0, len(conversations), MAX_MESSAGES_PER_FILE):
            chunk = conversations[i:i + MAX_MESSAGES_PER_FILE]
            if len(chunk) < MIN_MESSAGES_PER_FILE:
                continue
            out_path = f"{json_base_path}_{i}.json"
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump({"conversations": chunk}, out_f,
                          ensure_ascii=False, indent=4)
    else:
        # Single JSON file
        if not json_base_path.endswith(".json"):
            json_base_path += ".json"
        with open(json_base_path, "w", encoding="utf-8") as out_f:
            json.dump({"conversations": conversations}, out_f,
                      ensure_ascii=False, indent=4)


def convert_all_normalized_to_json(norm_folder, out_folder, whatsapp_name):
    """
    Process all normalized .txt files in `norm_folder` into JSON in `out_folder`.
    """
    os.makedirs(out_folder, exist_ok=True)

    for path in sorted(Path(norm_folder).glob("*.txt")):
        json_base = Path(out_folder) / Path(path.stem)
        txt_to_json_from_normalized(str(path), str(json_base), whatsapp_name)
        print(f"[OK] JSON created from: {path.name}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end WhatsApp preprocessing: "
            "raw exports (.txt) -> normalized text -> JSON for training."
        )
    )
    parser.add_argument(
        "whatsapp_name",
        type=str,
        help="Your WhatsApp display name (used to label your messages as 'gpt')."
    )
    args = parser.parse_args()

    if not args.whatsapp_name:
        print("Please provide your WhatsApp name.")
        raise SystemExit(1)

    print("=== STEP 1: Normalizing raw WhatsApp exports ===")
    normalize_all_chats(RAW_TXT_FOLDER, NORMALIZED_FOLDER)

    print("\n=== STEP 2: Converting normalized chats to JSON ===")
    convert_all_normalized_to_json(NORMALIZED_FOLDER, OUTPUT_JSON_FOLDER, args.whatsapp_name)

    print(f"\n Done. JSON files are in: {OUTPUT_JSON_FOLDER}")


if __name__ == "__main__":
    main()
