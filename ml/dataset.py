"""
CodeSheriff — Dataset Preparation Module

Loads the CodeSearchNet (Python) dataset from HuggingFace, applies heuristic
bug-type labelling, validates rows, and saves a processed CSV.

Usage (from project root):
    python -m ml.dataset              # original behavior
    python -m ml.dataset --balanced   # balanced pipeline with augmentation
"""

import argparse
import re
import sys
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `utils` is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import (
    SEED,
    DATA_PROCESSED_DIR,
    MAX_DATASET_SAMPLES,
    MAX_TOKEN_LENGTH,
    LABEL_NAMES,
    NUM_LABELS,
)
from utils.logger import get_logger

logger = get_logger("ml.dataset")

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Target distribution & validation thresholds
# ---------------------------------------------------------------------------
TARGET_COUNTS = {
    0: 3000,   # Clean
    1: 800,    # Null Reference Risk  (was 500 — more seeds now)
    2: 500,    # Type Mismatch        (was 300)
    3: 500,    # Security Vulnerability (was 300)
    4: 800,    # Logic Flaw           (was 500)
}

VALIDATION_THRESHOLDS = {
    0: 2000,   # Clean
    1: 600,    # Null Reference Risk
    2: 350,    # Type Mismatch
    3: 350,    # Security Vulnerability
    4: 600,    # Logic Flaw
}


# ---------------------------------------------------------------------------
# Seed templates for scarce classes
# ---------------------------------------------------------------------------
SECURITY_SEEDS = [
    # --- Original 10 ---
    'def get_user(uid):\n    query = "SELECT * FROM users WHERE id = " + uid\n    return db.execute(query)',
    "def search(term):\n    sql = f\"SELECT * FROM products WHERE name = '{term}'\"\n    cursor.execute(sql)",
    'def run_cmd(cmd):\n    os.system("ls " + cmd)',
    'def evaluate_expr(expr):\n    return eval(expr)',
    'def load_file(path):\n    return open(path).read()',
    'def execute(query, val):\n    cursor.execute("DELETE FROM logs WHERE id = " + val)',
    'def shell_run(user_cmd):\n    subprocess.run(user_cmd, shell=True)',
    'def get_config(key):\n    cmd = "cat /etc/config/" + key\n    os.system(cmd)',
    'def fetch(table, col, val):\n    q = "SELECT " + col + " FROM " + table + " WHERE x=" + val\n    return db.run(q)',
    "def render(template, data):\n    return eval(f\"f'{template}'\")",
    # --- Real-world CWE patterns ---
    'def update_user(uid, name):\n    sql = "UPDATE users SET name=\'%s\' WHERE id=%s" % (name, uid)\n    db.execute(sql)',
    'def delete_record(table, rid):\n    db.execute("DELETE FROM " + table + " WHERE id=" + str(rid))',
    'def read_user_file(filename):\n    with open("/data/" + filename) as f:\n        return f.read()',
    'def deserialize(data):\n    import pickle\n    return pickle.loads(data)',
    'def exec_code(code_str):\n    exec(code_str)',
    'def make_request(url):\n    import requests\n    return requests.get(url).text',
    'def log_query(user_input):\n    query = f"INSERT INTO logs (msg) VALUES (\'{user_input}\')"\n    db.execute(query)',
    'def render_page(template_str):\n    from jinja2 import Template\n    return Template(template_str).render()',
    'def run_script(script_name):\n    os.system("python " + script_name)',
    'def admin_action(action):\n    subprocess.Popen(action, shell=True)',
]

TYPE_MISMATCH_SEEDS = [
    # --- Original 10 ---
    'def greet(name, age):\n    return "Hello " + name + " you are " + age',
    'def calc(val):\n    if val = 10:\n        return True',
    'def check(score):\n    if score == "100":\n        return "perfect"',
    'def add_suffix(items, count):\n    return items + count',
    'def process(user_input):\n    total = 0 + user_input\n    return total',
    'def compare(a, b):\n    if str(a) == b + 1:\n        return True',
    'def build_msg(code):\n    msg = "Error code: " + code\n    return msg',
    'def validate(x):\n    if x = None:\n        raise ValueError("empty")',
    'def format_id(id):\n    return "ID-" + id',
    'def merge(text, num):\n    return text + num + " items"',
    # --- Real-world type confusion patterns ---
    'def display(status, count):\n    return "Status: " + status + ", Count: " + count',
    'def log_event(event, timestamp):\n    print("Event " + event + " at " + timestamp)',
    'def build_path(base, segment):\n    return base + "/" + segment + "/" + id',
    'def format_price(amount):\n    return "$" + amount',
    'def create_key(prefix, index):\n    return prefix + "_" + index',
    'def concat_results(header, data, footer):\n    return header + data + footer + 0',
    'def build_url(host, port):\n    return "http://" + host + ":" + port',
    'def render_row(name, score):\n    return "<tr><td>" + name + "</td><td>" + score + "</td></tr>"',
    'def label(category, num):\n    return category + ": " + num + " items"',
    'def summary(title, count, pct):\n    return title + " - " + count + " (" + pct + "%)"',
]

NULL_REFERENCE_SEEDS = [
    # --- Original 15 ---
    'def get_name(uid):\n    user = db.fetchone()\n    return user.name',
    'def load_config(path):\n    config = None\n    print(config.settings)',
    'def first_item(cursor):\n    row = cursor.fetchone()\n    return row["id"]',
    'def get_timeout(cfg):\n    timeout = cfg.get("timeout")\n    return int(timeout)',
    'def process_user(db, uid):\n    user = db.query(uid)\n    return user.email.lower()',
    'def read_setting(d, key):\n    val = d.get(key)\n    return val.strip()',
    'def parse_response(resp):\n    data = resp.json().get("result")\n    return data.value',
    'def find_order(db, oid):\n    order = db.find_one({"id": oid})\n    return order["total"]',
    'def get_header(headers, name):\n    h = headers.get(name)\n    return h.decode("utf-8")',
    'def first_match(results):\n    match = results.fetchone()\n    return match.group(0)',
    'def load_user(session):\n    user = session.get("user")\n    return user["name"]',
    'def get_parent(node):\n    parent = node.parent\n    return parent.value',
    'def extract_email(record):\n    email = record.get("email")\n    return email.split("@")[0]',
    'def get_price(product):\n    price = product.get("price")\n    return float(price)',
    'def read_env(key):\n    val = os.environ.get(key)\n    return val.upper()',
    # --- Chained attribute access (real-world) ---
    'def get_city(user):\n    profile = user.get("profile")\n    return profile["address"]["city"]',
    'def first_tag(post):\n    tags = post.get("tags")\n    return tags[0].lower()',
    'def author_name(book):\n    author = book.get("author")\n    return author.first_name + " " + author.last_name',
    # --- Optional / nullable return values ---
    'def get_manager(emp):\n    mgr = emp.manager\n    return mgr.email',
    'def next_task(queue):\n    task = queue.pop()\n    return task.priority',
    'def latest_log(db):\n    entry = db.logs.find_one(sort=[("ts", -1)])\n    return entry["message"]',
    # --- Iterator / generator exhaustion ---
    'def first_result(gen):\n    item = next(gen)\n    return item.value',
    'def read_line(fp):\n    line = fp.readline()\n    return line.strip().split(",")[0]',
    # --- Multi-step null chains ---
    'def process_payment(order):\n    card = order.get("payment")\n    last4 = card["number"][-4:]\n    return last4',
    'def resolve_host(config):\n    host = config.get("database", {}).get("host")\n    return host.split(":")[0]',
    'def get_token(headers):\n    auth = headers.get("Authorization")\n    return auth.replace("Bearer ", "")',
    # --- Real-world API patterns ---
    'def parse_webhook(data):\n    event = data.get("event")\n    return event["type"]',
    'def extract_id(response):\n    body = response.json()\n    return body["data"]["id"]',
    'def get_error_msg(resp):\n    err = resp.json().get("error")\n    return err["message"]',
    # --- Database cursor patterns ---
    'def fetch_username(cursor, uid):\n    cursor.execute("SELECT name FROM users WHERE id=%s", (uid,))\n    row = cursor.fetchone()\n    return row[0]',
    'def get_balance(conn, acct):\n    result = conn.execute("SELECT balance FROM accounts WHERE id=?", (acct,))\n    return result.fetchone()["balance"]',
    # --- None-returning function chains ---
    'def find_match(items, key):\n    match = None\n    for item in items:\n        if item.key == key:\n            match = item\n    return match.value',
    'def search(db, query):\n    result = db.search(query)\n    return result.title',
    'def pop_first(stack):\n    item = stack.pop(0) if stack else None\n    return item.name',
    'def get_conn(pool):\n    conn = pool.get_connection()\n    return conn.cursor()',
]

LOGIC_FLAW_SEEDS = [
    # --- Original 15 ---
    'def total(items):\n    s = 0\n    for i in range(len(items) + 1):\n        s += items[i]\n    return s',
    'def avg(values):\n    return sum(values) / len(values)',
    'def discount(price, pct):\n    return price / pct',
    'def check(status):\n    if status == "active" or "pending":\n        return True',
    'def last_elem(arr):\n    return arr[len(arr)]',
    'def split_bill(total, guests):\n    return total / len(guests)',
    'def is_valid(x):\n    if x == 0 or "null":\n        return False\n    return True',
    'def compute_rate(hits, misses):\n    return hits / (hits + misses)',
    'def normalize(vals):\n    mx = max(vals)\n    return [v / mx for v in vals]',
    'def convert(x, y):\n    if x == "high" or "critical":\n        return y * 2\n    return y',
    'def process_batch(items):\n    for i in range(len(items) + 1):\n        print(items[i])',
    'def ratio(a, b):\n    return a / b',
    'def mean_score(scores):\n    return sum(scores) / len(scores)',
    'def pct_change(old, new):\n    return (new - old) / old * 100',
    'def safe_div(a, b):\n    if b == 0 or None:\n        return 0\n    return a / b',
    # --- Real-world logic patterns ---
    'def index_of(arr, target):\n    for i in range(len(arr) + 1):\n        if arr[i] == target:\n            return i\n    return -1',
    'def clamp(val, lo, hi):\n    if val < lo or hi:\n        return lo\n    return val',
    'def median(data):\n    mid = len(data) / 2\n    return data[mid]',
    'def fibonacci(n):\n    if n == 0:\n        return 0\n    a, b = 0, 1\n    for _ in range(n + 1):\n        a, b = b, a + b\n    return a',
    'def deduplicate(items):\n    seen = set()\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n    return items',
    'def rotate(arr, k):\n    return arr[k:] + arr[:k+1]',
    'def power(base, exp):\n    result = 0\n    for _ in range(exp):\n        result *= base\n    return result',
    'def flatten(nested):\n    result = []\n    for sub in nested:\n        result.append(sub)\n    return result',
    'def is_palindrome(s):\n    return s == s[::-1].lower()',
    'def max_profit(prices):\n    profit = 0\n    for i in range(len(prices)):\n        for j in range(i, len(prices)):\n            profit = max(profit, prices[j] - prices[i])\n    return profit',
]


# ---------------------------------------------------------------------------
# Heuristic labelling functions (extended)
# ---------------------------------------------------------------------------

def _has_sql_injection_risk(code: str) -> bool:
    """Check for naive string concatenation in SQL-like patterns."""
    patterns = [
        r"""['"]SELECT\s.*?\+\s""",
        r"""['"]INSERT\s.*?\+\s""",
        r"""['"]UPDATE\s.*?\+\s""",
        r"""['"]DELETE\s.*?\+\s""",
        r"""f['"]\s*SELECT""",
        r"""\.format\(.*?SELECT""",
    ]
    return any(re.search(p, code, re.IGNORECASE) for p in patterns)


def _has_security_risk(code: str) -> bool:
    """Extended security vulnerability detection."""
    if _has_sql_injection_risk(code):
        return True
    patterns = [
        r"\bcursor\.execute\s*\([^)]*\+",             # cursor.execute(...+
        r"\bcursor\.execute\s*\([^)]*\.format\(",      # cursor.execute(....format(
        r"\bcursor\.execute\s*\(\s*f['\"]",            # cursor.execute(f"...
        r"\beval\s*\(",                                 # eval(
        r"\bos\.system\s*\([^)]*\+",                   # os.system(... +
        r"\bos\.system\s*\(\s*f['\"]",                 # os.system(f"...
        r"\bsubprocess\.\w+\([^)]*shell\s*=\s*True",  # subprocess with shell=True
        r"\bopen\s*\(\s*\w+\s*\)",                     # open(variable) without sanitization
    ]
    return any(re.search(p, code, re.IGNORECASE) for p in patterns)


def _has_null_reference_risk(code: str) -> bool:
    """Extended null-dereference detection."""
    patterns = [
        r"=\s*None[\s\S]{0,80}\.\w+",            # var = None ... var.something
        r"\.fetchone\(\)\.\w+",                    # result.fetchone().attr
        r"return\s+\w+\[.*?\]\.\w+",              # return x[...].attr
        r"\.get\s*\([^)]+\)\.\w+",                # .get(...).attr without fallback
        r"=\s*None\b[\s\S]{0,120}(?!is None)\.\w+\(",  # None assigned, method called
    ]
    return any(re.search(p, code) for p in patterns)


def _has_type_mismatch(code: str) -> bool:
    """Extended type-error detection."""
    patterns = [
        r"\bif\s+\w+\s*=[^=]",                     # single = in if condition
        r"\bint\s*\+\s*str\b",
        r"\bstr\s*\+\s*int\b",
        r'"\s*\+\s*\w+(?!\s*\()',                  # "string" + variable (likely int)
        r"\bstr\s*\(\s*\)\s*==\s*\d",              # str() == number
    ]
    return any(re.search(p, code) for p in patterns)


def _has_logic_flaw(code: str) -> bool:
    """Extended logic mistake detection."""
    patterns = [
        r"range\(len\(\w+\)\s*\+\s*1\)",           # off-by-one
        r"/\s*0\b",                                  # division by zero
        r"while\s+True\b(?![\s\S]{0,200}break)",   # infinite loop
        r"==\s*True\s*:",                            # redundant truth comparison
        r"\[\s*len\(\w+\)\s*\]",                    # index at len(x)
    ]
    return any(re.search(p, code) for p in patterns)


def assign_label(code: str) -> int:
    """
    Return a heuristic label for a code snippet.

    Priority order (first match wins):
        3  Security Vulnerability
        1  Null Reference Risk
        2  Type Mismatch
        4  Logic Flaw
        0  Clean
    """
    if _has_security_risk(code):
        return 3
    if _has_null_reference_risk(code):
        return 1
    if _has_type_mismatch(code):
        return 2
    if _has_logic_flaw(code):
        return 4
    return 0


# ---------------------------------------------------------------------------
# Code augmentation
# ---------------------------------------------------------------------------
RENAME_MAP = {
    "user":    ["user_id", "usr", "u"],
    "result":  ["res", "output", "ret"],
    "items":   ["arr", "elements", "lst"],
    "query":   ["sql", "q", "stmt"],
    "data":    ["payload", "info", "content"],
    "value":   ["val", "v", "x"],
    "input":   ["user_input", "raw", "inp"],
    "config":  ["cfg", "conf", "settings"],
    "path":    ["filepath", "fpath", "p"],
    "command": ["cmd", "shell_cmd", "c"],
}

_COMMENTS = [
    "# utility function",
    "# helper",
    "# process data",
    "# handler",
    "# internal",
]


def augment_code(code: str) -> List[str]:
    """Generate exactly 4 syntactic variants of a code snippet."""
    variants: List[str] = []

    for i in range(4):
        v = code

        # Variable renaming: pick 2-3 keys and rename
        rename_keys = [k for k in RENAME_MAP if re.search(r'\b' + k + r'\b', v)]
        num_renames = min(len(rename_keys), random.randint(2, 3))
        chosen_keys = random.sample(rename_keys, num_renames) if rename_keys else []
        for key in chosen_keys:
            replacement = random.choice(RENAME_MAP[key])
            v = re.sub(r'\b' + key + r'\b', replacement, v)

        # Whitespace: add blank line between statements (variant 0, 2)
        if i % 2 == 0:
            v = re.sub(r'\n(\s*\S)', r'\n\n\1', v, count=1)

        # Comment injection (variant 1, 3)
        if i % 2 == 1:
            comment = random.choice(_COMMENTS)
            v = comment + "\n" + v

        # Wrapper injection (50% — variants 2, 3)
        if i >= 2:
            indented = "\n".join("    " + line for line in v.splitlines())
            v = "def wrapper():\n" + indented

        variants.append(v)

    return variants


# ---------------------------------------------------------------------------
# Class-balanced sampling
# ---------------------------------------------------------------------------

def balance_dataset(df: pd.DataFrame, target_counts: dict) -> pd.DataFrame:
    """Resample each class to match target_counts using down/up-sampling + augmentation."""
    balanced = []
    for label, target in target_counts.items():
        class_df = df[df["label"] == label]
        if len(class_df) >= target:
            balanced.append(class_df.sample(target, random_state=SEED))
        else:
            balanced.append(class_df)
            needed = target - len(class_df)
            augmented = []
            pool = class_df["code"].tolist()
            if not pool:
                logger.warning(f"No samples for label {label} ({LABEL_NAMES[label]}) — cannot augment.")
                continue
            while len(augmented) < needed:
                source = random.choice(pool)
                for v in augment_code(source):
                    if len(augmented) < needed:
                        augmented.append({"code": v, "label": label})
            balanced.append(pd.DataFrame(augmented))

    result = pd.concat(balanced, ignore_index=True)
    result = result.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _validate_distribution(df: pd.DataFrame) -> None:
    """Warn if any class falls below its threshold."""
    for label, threshold in VALIDATION_THRESHOLDS.items():
        count = len(df[df["label"] == label])
        if count < threshold:
            logger.warning(
                f"Class '{LABEL_NAMES[label]}' has only {count} samples "
                f"(threshold: {threshold}). Consider adding more seed templates for this class."
            )


# ---------------------------------------------------------------------------
# Original pipeline (unchanged)
# ---------------------------------------------------------------------------

def prepare_dataset() -> pd.DataFrame:
    """Load, label, validate, and persist the dataset."""

    logger.info("Loading CodeSearchNet (Python split) from HuggingFace …")
    ds = load_dataset("code_search_net", "python", split="train")
    logger.info(f"Raw dataset size: {len(ds):,} samples")

    # Sub-sample for CPU feasibility
    if len(ds) > MAX_DATASET_SAMPLES:
        indices = random.sample(range(len(ds)), MAX_DATASET_SAMPLES)
        ds = ds.select(indices)
        logger.info(f"Sub-sampled to {len(ds):,} samples")

    # Extract code strings
    records = []
    for row in ds:
        code = row.get("func_code_string") or row.get("whole_func_string", "")
        if not code or not isinstance(code, str):
            continue
        # Skip if code is too long (heuristic: each char ≈ 0.3 tokens on average)
        if len(code) > MAX_TOKEN_LENGTH * 4:
            continue
        label = assign_label(code)
        records.append({"code": code, "label": label})

    df = pd.DataFrame(records)

    # Drop duplicates and any remaining NaN
    df.drop_duplicates(subset=["code"], inplace=True)
    df.dropna(inplace=True)

    logger.info(f"Processed dataset size: {len(df):,} samples")
    logger.info("Class distribution:")
    for label_id, label_name in LABEL_NAMES.items():
        count = len(df[df["label"] == label_id])
        pct = count / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  {label_id} ({label_name}): {count:,}  ({pct:.1f}%)")

    # Persist
    out_path = DATA_PROCESSED_DIR / "labeled_dataset.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved labeled dataset to {out_path}")

    return df


# ---------------------------------------------------------------------------
# Balanced pipeline
# ---------------------------------------------------------------------------

def prepare_balanced_dataset() -> pd.DataFrame:
    """Load, label, inject seeds, balance, split, and persist."""
    from sklearn.model_selection import train_test_split

    # ---- Step 1: Load and label CodeSearchNet samples ---------------------
    logger.info("Loading CodeSearchNet (Python split) from HuggingFace …")
    ds = load_dataset("code_search_net", "python", split="train")
    logger.info(f"Raw dataset size: {len(ds):,} samples")

    if len(ds) > MAX_DATASET_SAMPLES:
        indices = random.sample(range(len(ds)), MAX_DATASET_SAMPLES)
        ds = ds.select(indices)
        logger.info(f"Sub-sampled to {len(ds):,} samples")

    records = []
    for row in ds:
        code = row.get("func_code_string") or row.get("whole_func_string", "")
        if not code or not isinstance(code, str):
            continue
        if len(code) > MAX_TOKEN_LENGTH * 4:
            continue
        label = assign_label(code)
        records.append({"code": code, "label": label})

    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["code"], inplace=True)
    df.dropna(inplace=True)

    logger.info(f"Labeled {len(df):,} CodeSearchNet samples")
    logger.info("Pre-balance distribution:")
    for lid, lname in LABEL_NAMES.items():
        count = len(df[df["label"] == lid])
        logger.info(f"  {lid} ({lname}): {count:,}")

    # ---- Step 2: Inject seed templates ------------------------------------
    seed_records = []
    for code in SECURITY_SEEDS:
        seed_records.append({"code": code, "label": 3})
    for code in TYPE_MISMATCH_SEEDS:
        seed_records.append({"code": code, "label": 2})
    for code in NULL_REFERENCE_SEEDS:
        seed_records.append({"code": code, "label": 1})
    for code in LOGIC_FLAW_SEEDS:
        seed_records.append({"code": code, "label": 4})

    seed_df = pd.DataFrame(seed_records)
    df = pd.concat([df, seed_df], ignore_index=True)
    df.drop_duplicates(subset=["code"], inplace=True)
    logger.info(f"After seed injection: {len(df):,} samples")

    # ---- Step 3: Balance --------------------------------------------------
    logger.info("Balancing dataset …")
    balanced_df = balance_dataset(df, TARGET_COUNTS)

    # ---- Step 4: Validate -------------------------------------------------
    _validate_distribution(balanced_df)

    # ---- Step 5: Print distribution ---------------------------------------
    logger.info("── Dataset distribution after balancing ──────────────────")
    total = len(balanced_df)
    for lid in range(NUM_LABELS):
        count = len(balanced_df[balanced_df["label"] == lid])
        logger.info(f"  {LABEL_NAMES[lid]:<24}: {count}")
    logger.info("  ─────────────────────────────────────────────────────────")

    # ---- Step 6: Stratified split (70/15/15) ------------------------------
    codes = balanced_df["code"].tolist()
    labels = balanced_df["label"].tolist()

    train_codes, temp_codes, train_labels, temp_labels = train_test_split(
        codes, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_codes, test_codes, val_labels, test_labels = train_test_split(
        temp_codes, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
    )

    train_df = pd.DataFrame({"code": train_codes, "label": train_labels})
    val_df = pd.DataFrame({"code": val_codes, "label": val_labels})
    test_df = pd.DataFrame({"code": test_codes, "label": test_labels})

    logger.info(f"  Total samples        : {total}")
    logger.info(f"  Train / Val / Test   : {len(train_df)} / {len(val_df)} / {len(test_df)}")

    # ---- Step 7: Save -----------------------------------------------------
    balanced_df.to_csv(DATA_PROCESSED_DIR / "labeled_dataset_balanced.csv", index=False)
    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)

    logger.info(f"Saved balanced dataset → {DATA_PROCESSED_DIR / 'labeled_dataset_balanced.csv'}")
    logger.info(f"Saved train split      → {DATA_PROCESSED_DIR / 'train.csv'}")
    logger.info(f"Saved val split        → {DATA_PROCESSED_DIR / 'val.csv'}")
    logger.info(f"Saved test split       → {DATA_PROCESSED_DIR / 'test.csv'}")

    return balanced_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeSheriff dataset preparation")
    parser.add_argument("--balanced", action="store_true",
                        help="Run the balanced pipeline with augmentation and splits")
    args = parser.parse_args()

    if args.balanced:
        prepare_balanced_dataset()
    else:
        prepare_dataset()
