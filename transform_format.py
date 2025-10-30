import os
import re
import glob
import random
from datetime import datetime, timedelta
from pathlib import Path

# ==================== CONFIG ====================
INPUT_FOLDER = "data/unformated_text"  # source .txt (WhatsApp export format)
OUTPUT_FOLDER = "data/raw_data"       # transformed + augmented output
SOURCE_NAME_KEYS = ("random1", "random2")  # which files are your Random1/2 sources (case-insensitive)
GAP_THRESHOLD_MINUTES = 90            # only insert into gaps >= this length
BLOCK_LEN_MIN, BLOCK_LEN_MAX = 5, 6   # coherent block size from source chats
MIN_SPACING_SECONDS = 45              # spacing between messages inside inserted block
MAX_SPACING_SECONDS = 180
TARGET_INSERT_FRACTION = 0.20         # want ~20% of final messages to be inserted
SEED = 42
# =================================================

LINE_RE = re.compile(r'^(\d{2})/(\d{2})/(\d{4}), (\d{2}):(\d{2}) - ([^:]+): (.*)$')

def parse_chat_lines(path):
    """Parse WhatsApp export into list of {dt,name,text}. Attach multiline continuations to previous message."""
    msgs = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            m = LINE_RE.match(raw)
            if m:
                d, mo, y, hh, mm, name, text = m.groups()
                dt = datetime(int(y), int(mo), int(d), int(hh), int(mm))
                msgs.append({"dt": dt, "name": name, "text": text})
            else:
                # continuation / system; glue to previous if present
                if msgs and raw.strip():
                    msgs[-1]["text"] = (msgs[-1]["text"] + "\n" + raw.strip()).strip()
                # else ignore stray lines
    msgs.sort(key=lambda x: x["dt"])
    return msgs

def format_whatsapp_line(msg):
    return f'{msg["dt"].strftime("%d/%m/%Y, %H:%M")} - {msg["name"]}: {msg["text"]}'

def format_transformed_line(msg):
    # to "[dd.mm.yy, HH:MM] Name: Message"
    new_date = msg["dt"].strftime("%d.%m.%y")
    return f'[{new_date}, {msg["dt"].strftime("%H:%M")}] {msg["name"]}: {msg["text"]}'

def build_source_blocks(source_paths):
    """Create a pool of coherent 5–6 message blocks from sources; ensure at least 2 speakers in a block."""
    blocks = []
    for sp in source_paths:
        msgs = parse_chat_lines(sp)
        if len(msgs) < BLOCK_LEN_MIN:
            continue
        i = 0
        while i + BLOCK_LEN_MIN <= len(msgs):
            L = random.randint(BLOCK_LEN_MIN, min(BLOCK_LEN_MAX, len(msgs) - i))
            block = msgs[i:i+L]
            # require at least two different speakers and at least one switch
            names = [m["name"] for m in block]
            if len(set(names)) >= 2 and any(names[j] != names[j-1] for j in range(1, len(names))):
                # strip dt (we’ll retime them)
                blocks.append([{"name": m["name"], "text": m["text"]} for m in block])
            # advance by 2–4 to diversify
            i += random.randint(2, 4)
    random.shuffle(blocks)
    return blocks

def find_gaps(target_msgs, threshold_minutes=GAP_THRESHOLD_MINUTES):
    gaps = []
    thresh = timedelta(minutes=threshold_minutes)
    for i in range(len(target_msgs) - 1):
        a = target_msgs[i]["dt"]
        b = target_msgs[i+1]["dt"]
        if b - a >= thresh:
            gaps.append((i, i+1, a, b))
    # sort by gap length descending so we use the roomiest first
    gaps.sort(key=lambda g: (g[3] - g[2]), reverse=True)
    return gaps

def retime_block_into_gap(block, gap_start, gap_end):
    """Retime a block to fit between (gap_start, gap_end) with random spacing."""
    margin = timedelta(minutes=3)
    spacings = [timedelta(seconds=random.randint(MIN_SPACING_SECONDS, MAX_SPACING_SECONDS))
                for _ in range(len(block)-1)]
    total_span = sum(spacings, timedelta(0))
    earliest = gap_start + margin
    latest = gap_end - margin - total_span
    if latest <= earliest:
        return None
    start = earliest + (latest - earliest) * random.random()
    t = start
    retimed = [{"dt": t, "name": block[0]["name"], "text": block[0]["text"]}]
    for i in range(1, len(block)):
        t = t + spacings[i-1]
        retimed.append({"dt": t, "name": block[i]["name"], "text": block[i]["text"]})
    if retimed[-1]["dt"] >= gap_end - margin:
        return None
    return retimed

def compute_insert_target_count(n_original, target_fraction=TARGET_INSERT_FRACTION):
    """
    We want inserted / final ~= target_fraction
    Let M be inserted, N be original. M / (N + M) = f  =>  M = f*N / (1 - f)
    """
    if target_fraction <= 0 or target_fraction >= 0.95:
        return 0
    M = target_fraction * n_original / (1.0 - target_fraction)
    return max(0, int(round(M)))

def augment_messages_with_blocks(target_msgs, blocks, target_insert_count):
    """Insert blocks into gaps until we reach ~target_insert_count messages or run out of room."""
    msgs = list(target_msgs)
    inserted = 0
    gaps = find_gaps(msgs, GAP_THRESHOLD_MINUTES)

    for (i_before, i_after, gstart, gend) in gaps:
        if inserted >= target_insert_count or not blocks:
            break
        # try a handful of blocks to fit this gap
        tried = 0
        while blocks and tried < 10 and inserted < target_insert_count:
            block = blocks.pop(0)
            retimed = retime_block_into_gap(block, gstart, gend)
            if retimed is None:
                # put it back to try on another gap later
                blocks.append(block)
                tried += 1
                continue
            msgs.extend(retimed)
            msgs.sort(key=lambda x: x["dt"])
            inserted += len(retimed)
            break
    return msgs, inserted

def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # collect all input .txt
    all_txts = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.txt")))
    # identify Random1/2 sources
    sources = [p for p in all_txts if any(k in Path(p).name.lower() for k in SOURCE_NAME_KEYS)]
    if not sources:
        print("WARNING: no Random1/Random2 sources found. Nothing will be inserted.")
    source_blocks = build_source_blocks(sources) if sources else []

    for in_path in all_txts:
        fname = Path(in_path).name
        # Skip augmenting the source chats themselves; just transform them
        is_source = any(k in fname.lower() for k in SOURCE_NAME_KEYS)

        target_msgs = parse_chat_lines(in_path)

        if not is_source and source_blocks and target_msgs:
            # how many messages should we insert?
            insert_target = compute_insert_target_count(len(target_msgs), TARGET_INSERT_FRACTION)
            # Make a working copy of blocks so each target gets a fresh shuffle
            blocks_pool = source_blocks.copy()
            random.shuffle(blocks_pool)
            augmented, inserted = augment_messages_with_blocks(target_msgs, blocks_pool, insert_target)
            msgs_out = augmented
            print(f"{fname}: inserted {inserted} messages (~{TARGET_INSERT_FRACTION*100:.0f}%)")
        else:
            msgs_out = target_msgs
            if is_source:
                print(f"{fname}: source chat; no augmentation (just transform).")

        # write transformed format to OUTPUT_FOLDER
        out_path = os.path.join(OUTPUT_FOLDER, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            for m in msgs_out:
                f.write(format_transformed_line(m) + "\n")

    print(f"Done. Transformed (and augmented) files saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
