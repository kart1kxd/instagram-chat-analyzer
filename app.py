"""
Instagram Chat Analyzer — Industry‑grade Streamlit App (single‑file)

What’s new (major upgrade)
- Robust encoding & normalization to fix mojibake, weird names, and control chars
- Big‑data friendly loading: ZIP, multi‑JSON, or direct **Local folder path** (no upload)
- Sidebar **Settings** with: date range filter, Hinglish/English stopwords, custom stopwords, token rules
- Tabs UI: Overview · Words · Emojis · Timing · Media · Search · Export
- Better analytics: n‑grams (bigrams), per‑sender message length, starters by day, longest gaps, streaks
- Word frequencies with participant aliases and clean columns
- Message search with filters (sender, contains text, date window)
- Exports: per‑conversation CSV, analytics JSON, word usage CSV
- Safer defaults, caching, and clearer error messages

Run:
  pip install streamlit pandas numpy matplotlib emoji tzdata
  # optional (for wordclouds & Hindi/Indic tokenization you'd add extras later)
  streamlit run app.py
"""

from __future__ import annotations
import io
import os
import re
import json
import zipfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import emoji as emoji_lib  # optional, improves emoji detection
except Exception:
    emoji_lib = None

# ---------------------------
# Stopwords & text helpers
# ---------------------------
BASIC_STOPWORDS = set(
    '''a an and are as at be by for from has have i if in into is it its of on or
    s so t that the their them then there these they this to u up was we were what
    when where which who why will with you your yours me my mine our ours yours
    he she him her his hers it's im i'm you're were we're they're ive i've you've we've they’ve
    just not don dont didn didn't wont won't cant can't couldnt should shouldn't would wouldn't
    do does did doing done also very much than too via rt ok okay hmm uh um yo hi hello hey
    '''.split()
)

HINGLISH_STOPWORDS = set('''
    hai ho hun hua hue hui he hain hogi hoga hoge tha the thi thay thiye
    mai main me mein mera meri mere tum tu aap ap apka apki apke aapka aapki aapke
    tumhara tumhari tumhare tera teri tere uska uski uske unka unki unke hum ham hamara hamari hamare
    kuch sab yeh ye yah yaha yahan waha wahan woh wo jo jis jisko jisse jise
    kya kyun kyu kyon kyunki kyuki kyo
    se ko ka ki ke toh to ho gya gaya gyi gayi gaye rha rhi rhe raha rahe rahi
    nhi nahi na han haan haanji ji bhai bro sis madam sir bhaiya behen
'''.split())

# Common Instagram "system" tokens we don't want to count as words
SYSTEM_TOKENS = set(
    ['sent', 'attachm', 'attachment', 'omitted', 'image', 'photo', 'video',
     'sticker', 'gif', 'shared', 'reacted', 'audio', 'call', 'missed', 'voice']
)

APOSTROPHE_CHARS = "’`´ʻʼʹˈ‛❜"

@dataclass
class Thread:
    title: str
    participants: List[str]
    df: pd.DataFrame  # columns: sender, ts, text, has_* flags, emojis, n_chars, n_words

# --- String normalization to fix mojibake and weird encodings ---
def _maybe_demojibake(s: str) -> str:
    """Try to reverse common UTF-8/Latin-1 mojibake like 'AÃ¡' -> 'Á'."""
    try:
        if any(ch in s for ch in "ÃÂ¢ð£¨"):  # heuristic
            return s.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
    except Exception:
        pass
    return s


def normalize_str(s: str) -> str:
    if s is None:
        return ''
    s = _maybe_demojibake(str(s))
    s = unicodedata.normalize('NFKC', s)
    # strip control characters / non-printables
    s = ''.join(ch for ch in s if ch.isprintable())
    return s.strip()


# ---------------------------
# JSON readers
# ---------------------------

def _read_json_bytes(b: bytes):
    # Prefer utf-8 with BOM handling; then utf-16 variants; finally latin-1
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return json.loads(b.decode(enc))
        except Exception:
            continue
    # Fallback: best-effort ignore errors
    return json.loads(b.decode("utf-8", errors="ignore"))


def _iter_message_jsons_from_zip(zf: zipfile.ZipFile):
    # Instagram messages typically live under messages/inbox/<thread>/message_*.json
    for name in zf.namelist():
        base = os.path.basename(name)
        if base.startswith("message_") and base.endswith(".json") and "/messages/inbox/" in name:
            with zf.open(name) as f:
                yield name, _read_json_bytes(f.read())


def _iter_message_jsons_from_files(files):
    for f in files:
        if f.name.endswith('.json') and getattr(f, 'size', 1) > 0:
            yield f.name, _read_json_bytes(f.getvalue())


# ---------------------------
# Text/emoji utils
# ---------------------------

def _extract_emojis(s: str):
    if not s:
        return []
    if emoji_lib:
        return [ch for ch in s if ch in emoji_lib.EMOJI_DATA]
    return re.findall(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]", s)


def _clean_text(s: str):
    if s is None:
        return ""
    s = re.sub(f"[{APOSTROPHE_CHARS}]", "'", s)
    s = re.sub(r"https?://\S+", " ", s)        # strip URLs
    s = re.sub(r"[^\w#@' ]+", " ", s)          # keep words/hashtags/mentions
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_words(s: str, stop_extra: Iterable[str] = ()): 
    s = _clean_text(s)
    if not s:
        return []
    stop = BASIC_STOPWORDS | set(stop_extra) | SYSTEM_TOKENS
    tokens = s.split()
    out = []
    for tok in tokens:
        if tok == "'":
            continue
        if tok.endswith("'s"):
            tok = tok[:-2]
        if tok in stop:
            continue
        # Drop very short tokens unless hashtag/mention (gets rid of mojibake singletons like 'ð', 'à')
        if (len(tok) <= 1) and not (tok.startswith('#') or tok.startswith('@')):
            continue
        if tok.isdigit():
            continue
        out.append(tok)
    return out


def _bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]


# ---------------------------
# Thread parsing
# ---------------------------

def _parse_one_thread(thread_obj):
    participants = [normalize_str(p.get('name', '')) for p in thread_obj.get('participants', [])]
    title = normalize_str(thread_obj.get('title')) or ", ".join(participants) or "(untitled)"
    messages = thread_obj.get('messages', [])
    rows = []
    for m in messages:
        sender = normalize_str(m.get('sender_name') or m.get('sender') or 'Unknown')
        ts_ms = m.get('timestamp_ms')
        if ts_ms is None:
            ts_ms = m.get('timestamp')
            if isinstance(ts_ms, (int, float)) and ts_ms < 10**12:
                ts_ms *= 1000
        try:
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else None
        except Exception:
            ts = None
        text = normalize_str(m.get('content'))
        has_photo = bool(m.get('photos'))
        has_video = bool(m.get('videos'))
        has_share = bool(m.get('share'))
        has_sticker = bool(m.get('sticker'))
        emojis = _extract_emojis(text) if text else []
        rows.append({
            'sender': sender,
            'ts': ts,
            'text': text or "",
            'has_photo': has_photo,
            'has_video': has_video,
            'has_share': has_share,
            'has_sticker': has_sticker,
            'emojis': emojis,
            'n_chars': len(text) if text else 0,
            'n_words': len(_tokenize_words(text)) if text else 0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['sender'] = df['sender'].apply(normalize_str)
        df = df.sort_values('ts').reset_index(drop=True)
    return Thread(title=title, participants=participants, df=df)


def _collect_threads_from_json_objs(json_objs: List[Tuple[str, dict]]):
    threads = []
    for name, obj in json_objs:
        if isinstance(obj, dict) and 'messages' in obj and 'participants' in obj:
            threads.append(_parse_one_thread(obj))
        elif isinstance(obj, dict) and 'conversations' in obj:
            for t in obj['conversations']:
                threads.append(_parse_one_thread(t))
        elif isinstance(obj, list):
            for t in obj:
                if isinstance(t, dict) and 'messages' in t:
                    threads.append(_parse_one_thread(t))
    # merge same-title threads (message_1.json, message_2.json...)
    merged = {}
    for th in threads:
        merged.setdefault(th.title, []).append(th.df)
    out: List[Thread] = []
    for title, dfs in merged.items():
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        df = df.sort_values('ts').reset_index(drop=True) if not df.empty else df
        participants = sorted(list(set(df['sender'].dropna().unique()))) if not df.empty else []
        out.append(Thread(title=title, participants=participants, df=df))
    out.sort(key=lambda t: (t.df['ts'].max() if not t.df.empty else datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return out


# ---------------------------
# Analytics helpers
# ---------------------------

def get_high_level_stats(df: pd.DataFrame):
    if df.empty:
        return {}
    total = len(df)
    by_sender = df['sender'].value_counts().to_dict()
    first = df['ts'].min()
    last = df['ts'].max()
    days = max(1, (last - first).days or 1)
    msgs_per_day = round(total / days, 2) if days else total
    media_counts = {
        'photos': int(df['has_photo'].sum()),
        'videos': int(df['has_video'].sum()),
        'shares/links': int(df['has_share'].sum()),
        'stickers': int(df['has_sticker'].sum()),
    }
    avg_len_chars = df.groupby('sender')['n_chars'].mean().round(1).to_dict()
    avg_len_words = df.groupby('sender')['n_words'].mean().round(1).to_dict()
    return {
        'total': total,
        'by_sender': by_sender,
        'first': first,
        'last': last,
        'msgs_per_day': msgs_per_day,
        'media_counts': media_counts,
        'avg_len_chars': avg_len_chars,
        'avg_len_words': avg_len_words,
    }


def build_word_frequencies(
    df: pd.DataFrame, top_n=30, stop_extra: Iterable[str] = (), bigrams=False
):
    if df.empty:
        return pd.DataFrame(), {}
    overall_counter = Counter()
    sender_counters = defaultdict(Counter)
    for _, row in df.iterrows():
        toks = _tokenize_words(row['text'], stop_extra)
        if not toks:
            continue
        if bigrams:
            toks = _bigrams(toks)
        overall_counter.update(toks)
        sender_counters[row['sender']].update(toks)
    top = overall_counter.most_common(top_n)
    words = [w for w, _ in top]
    data = {'Term': words, 'Total (all participants)': [overall_counter[w] for w in words]}
    for sender, cnt in sender_counters.items():
        data[f"{sender}"] = [cnt.get(w, 0) for w in words]
    dfw = pd.DataFrame(data)
    shares = {}
    for w in words:
        total = sum(sender_counters[s].get(w, 0) for s in sender_counters)
        if total == 0:
            continue
        shares[w] = {s: sender_counters[s].get(w, 0) / total for s in sender_counters}
    return dfw, shares


def build_emoji_frequencies(df: pd.DataFrame, top_n=20):
    if df.empty:
        return pd.DataFrame()
    overall = Counter()
    by_sender = defaultdict(Counter)
    for _, row in df.iterrows():
        if not row['emojis']:
            continue
        overall.update(row['emojis'])
        by_sender[row['sender']].update(row['emojis'])
    top = overall.most_common(top_n)
    emojis = [e for e, _ in top]
    data = {'emoji': emojis, 'Total': [overall[e] for e in emojis]}
    for sender, cnt in by_sender.items():
        data[sender] = [cnt.get(e, 0) for e in emojis]
    return pd.DataFrame(data)


def build_response_times(df: pd.DataFrame):
    if df.empty:
        return {}, {}
    resp = defaultdict(list)
    ts = df['ts'].tolist()
    snd = df['sender'].tolist()
    for i in range(1, len(df)):
        if snd[i] != snd[i-1] and ts[i] and ts[i-1]:
            delta = (ts[i] - ts[i-1]).total_seconds()
            if 0 < delta <= 3 * 24 * 3600:  # cap at 3 days
                resp[snd[i]].append(delta)
    summary = {}
    for s, vals in resp.items():
        if vals:
            arr = np.array(vals)
            summary[s] = {
                'count': int(len(vals)),
                'avg_sec': float(arr.mean()),
                'median_sec': float(np.median(arr)),
                'p90_sec': float(np.percentile(arr, 90)),
            }
    return resp, summary


def daily_activity(df: pd.DataFrame):
    if df.empty:
        return pd.Series(dtype=int)
    return df.set_index('ts').resample('D').size()


def heatmap_hour_weekday(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp['weekday'] = tmp['ts'].dt.weekday  # 0=Mon
    tmp['hour'] = tmp['ts'].dt.hour
    return tmp.pivot_table(index='weekday', columns='hour', values='text', aggfunc='count', fill_value=0)


def starters_by_day(df: pd.DataFrame):
    if df.empty:
        return {}
    day_first = df.copy()
    day_first['date'] = day_first['ts'].dt.floor('D')
    firsts = day_first.groupby('date').first()['sender']
    return firsts.value_counts().to_dict()


def longest_gap(df: pd.DataFrame):
    if df.empty:
        return 0, None, None
    t = df['ts'].dropna().sort_values().values
    gaps = np.diff(t).astype('timedelta64[s]').astype(int)
    if len(gaps) == 0:
        return 0, None, None
    i = int(np.argmax(gaps))
    gap_sec = int(gaps[i])
    return gap_sec, pd.Timestamp(t[i]).to_pydatetime(), pd.Timestamp(t[i+1]).to_pydatetime()


def day_streak(df: pd.DataFrame):
    if df.empty:
        return 0
    days = pd.to_datetime(df['ts'].dt.date.unique())
    days = np.sort(days.values)
    if len(days) == 0:
        return 0
    streak = best = 1
    for i in range(1, len(days)):
        if (days[i] - days[i-1]).astype('timedelta64[D]').astype(int) == 1:
            streak += 1
            best = max(best, streak)
        else:
            streak = 1
    return int(best)


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Instagram Chat Analyzer", layout="wide")
st.title("📊 Instagram Chat Analyzer (Pro)")

st.markdown(
    "Upload your Instagram export (.zip) **or** multiple `message_*.json` files from `messages/inbox/...`  \n"
    "**Tip for big exports (GBs):** Unzip locally and either upload only the JSONs you need or provide a folder path."
)

col_zip, col_json = st.columns(2)
zip_file = col_zip.file_uploader(
    "Upload Instagram export .zip (may be large)", type=["zip"], accept_multiple_files=False,
    help="If your ZIP exceeds server limits, unzip locally and use the JSON or folder path options."
)
json_files = col_json.file_uploader(
    "Or upload multiple message_*.json files", type=["json"], accept_multiple_files=True,
    help="Drag & drop many JSONs at once."
)

st.markdown(":grey[Running locally with a giant export? Paste your folder path below to load JSON files directly, no upload required.]")
local_folder = st.text_input(
    "Local folder path to your `messages/inbox` (optional)", value="",
    placeholder=r"C:\\Users\\you\\Downloads\\instagram-data\\messages\\inbox"
)
use_local_btn = st.button("Load from local folder", type="secondary")

@st.cache_data(show_spinner=True)
def load_threads_from_inputs(zip_bytes, json_files_in):
    json_objs: List[Tuple[str, dict]] = []
    if zip_bytes is not None:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name, obj in _iter_message_jsons_from_zip(zf):
                json_objs.append((name, obj))
    if json_files_in:
        for name, obj in _iter_message_jsons_from_files(json_files_in):
            json_objs.append((name, obj))
    return _collect_threads_from_json_objs(json_objs)

@st.cache_data(show_spinner=True)
def load_threads_from_folder(folder_path: str):
    json_objs: List[Tuple[str, dict]] = []
    if not folder_path:
        return []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.startswith("message_") and fname.endswith(".json"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "rb") as f:
                        json_objs.append((fpath, _read_json_bytes(f.read())))
                except Exception as e:
                    st.warning(f"Couldn't read {fpath}: {e}")
    return _collect_threads_from_json_objs(json_objs)

threads: List[Thread] = []
if use_local_btn and local_folder.strip():
    with st.spinner("Scanning local folder for message_*.json ..."):
        threads = load_threads_from_folder(local_folder.strip())
elif zip_file is not None or (json_files and len(json_files) > 0):
    threads = load_threads_from_inputs(zip_file.getvalue() if zip_file else None, json_files)

if not threads:
    st.info("👈 Upload or load your data to begin.")
    st.stop()

st.success(f"Loaded {len(threads)} conversations.")

# ---------------------------
# Sidebar Settings
# ---------------------------
with st.sidebar:
    st.header("Settings")

    # Participant aliasing
    st.subheader("Participant Aliases")
    all_names = sorted(list({p for t in threads for p in t.participants} | {n for t in threads for n in t.df['sender'].unique()}))
    alias_help = "Optional: rename participants for cleaner column labels (format: original => alias per line)."
    default_alias_text = "\n".join([f"{n} => {n}" for n in all_names[:10]]) if all_names else ""
    alias_text = st.text_area("Aliases", value=default_alias_text, height=150, help=alias_help)

    alias_map: Dict[str, str] = {}
    for line in alias_text.splitlines():
        if '=>' in line:
            orig, alias = [x.strip() for x in line.split('=>', 1)]
            if orig:
                alias_map[orig] = alias or orig

    # Date range filter
    st.subheader("Date Range")
    all_ts = pd.concat([t.df['ts'] for t in threads if not t.df.empty])
    min_dt = (all_ts.min() or datetime.now(timezone.utc)).to_pydatetime()
    max_dt = (all_ts.max() or datetime.now(timezone.utc)).to_pydatetime()
    date_range = st.date_input("Limit to dates (inclusive)", value=(min_dt.date(), max_dt.date()))

    # Stopword controls
    st.subheader("Text & Tokens")
    use_hinglish = st.checkbox("Add Hinglish stopwords", value=True)
    drop_short = st.checkbox("Drop 1‑char tokens (except #/@)", value=True)
    drop_system = st.checkbox("Drop Instagram system tokens (sent/attachm/…)", value=True)
    custom_sw = st.text_area("Custom stopwords (space or newline separated)", "")

    # N‑gram option
    use_bigrams = st.checkbox("Use bigrams (two‑word phrases)", value=False)

# helper to apply alias map

def apply_alias(name: str) -> str:
    return alias_map.get(name, name)

# ---------------------------
# Choose conversation
# ---------------------------
thread_titles = [f"{t.title} — ({len(t.df)} msgs)" for t in threads]
selected = st.selectbox("Choose a conversation", options=range(len(threads)), format_func=lambda i: thread_titles[i])
thread = threads[selected]

# Apply date filter
if thread.df.empty:
    st.warning("This conversation has no messages to analyze.")
    st.stop()

start_date, end_date = date_range if isinstance(date_range, tuple) else (date_range, date_range)
start_ts = pd.Timestamp(start_date).tz_localize('UTC')
end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize('UTC')  # inclusive end
mask = (thread.df['ts'] >= start_ts) & (thread.df['ts'] < end_ts)

df_view = thread.df.loc[mask].copy()

# Apply aliases
if not df_view.empty:
    df_view['sender'] = df_view['sender'].map(apply_alias)

# Build stopword set
extra_sw = set()
if use_hinglish:
    extra_sw |= HINGLISH_STOPWORDS
if drop_system:
    extra_sw |= SYSTEM_TOKENS
if custom_sw.strip():
    extra_sw |= set(re.split(r"\s+", custom_sw.strip()))

# ---------------------------
# Tabs
# ---------------------------

tab_overview, tab_words, tab_emojis, tab_timing, tab_media, tab_search, tab_export = st.tabs([
    "Overview", "Words", "Emojis", "Timing", "Media", "Search", "Export"
])

# ===== Overview =====
with tab_overview:
    st.subheader(f"💬 Conversation: {thread.title}")
    if thread.participants:
        st.caption("Participants: " + ", ".join([apply_alias(p) for p in thread.participants]))

    stats = get_high_level_stats(df_view)
    if not stats:
        st.info("No messages in selected date range.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total messages", stats['total'])
        c2.metric("Messages/day (avg)", stats['msgs_per_day'])
        c3.metric("First message", stats['first'].strftime('%Y-%m-%d'))
        c4.metric("Last message", stats['last'].strftime('%Y-%m-%d'))

        st.write("**Messages by sender**")
        fig, ax = plt.subplots()
        labels = [apply_alias(k) for k in stats['by_sender'].keys()]
        values = list(stats['by_sender'].values())
        ax.bar(labels, values)
        ax.set_ylabel('Messages')
        ax.set_xticklabels(labels, rotation=15, ha='right')
        st.pyplot(fig, transparent=True)

        # Starters, longest gap, streak
        start_counts = starters_by_day(df_view)
        gap_sec, gap_start, gap_end = longest_gap(df_view)
        streak_days = day_streak(df_view)
        c5, c6, c7 = st.columns(3)
        if start_counts:
            starter_name = apply_alias(max(start_counts, key=start_counts.get))
            c5.metric("Most day-starters", f"{starter_name}", help=str(start_counts))
        if gap_sec:
            c6.metric("Longest quiet gap", f"{round(gap_sec/3600,1)} h", help=f"{gap_start} → {gap_end}")
        c7.metric("Longest daily streak", f"{streak_days} days")

# ===== Words =====
with tab_words:
    st.subheader("Most used terms")
    top_n = st.slider("Top N", 10, 100, 30, 5)
    dfw, shares = build_word_frequencies(df_view, top_n=top_n, stop_extra=extra_sw, bigrams=use_bigrams)
    if dfw.empty:
        st.info("No text messages to compute frequencies.")
    else:
        st.dataframe(dfw, use_container_width=True)
        if shares:
            rows = []
            for w, sdict in shares.items():
                max_sender = max(sdict, key=sdict.get)
                rows.append({'Term': w,
             'Most used by': max_sender,
             'Share %': round(sdict[max_sender]*100, 1)})

            byword_df = pd.DataFrame(rows)
            byword_df['Most used by'] = byword_df['Most used by'].map(apply_alias)
            byword_df = byword_df.sort_values('Share %', ascending=False)
            st.caption("Who uses each term more (share among participants)")
            st.dataframe(byword_df, use_container_width=True)

# ===== Emojis =====
with tab_emojis:
    st.subheader("Emoji usage")
    top_e = st.slider("Top emojis", 5, 50, 20, 5)
    emoji_df = build_emoji_frequencies(df_view, top_n=top_e)
    if emoji_df.empty:
        st.info("No emojis detected.")
    else:
        st.dataframe(emoji_df, use_container_width=True)

# ===== Timing =====
with tab_timing:
    st.subheader("Activity over time")
    series = daily_activity(df_view)
    if series.empty:
        st.info("No timestamps available.")
    else:
        fig3, ax3 = plt.subplots()
        ax3.plot(series.index, series.values)
        ax3.set_ylabel('Messages per day')
        ax3.set_xlabel('Date')
        st.pyplot(fig3, transparent=True)

        st.caption("Activity heatmap (weekday vs hour)")
        pivot = heatmap_hour_weekday(df_view)
        if not pivot.empty:
            fig4, ax4 = plt.subplots()
            im = ax4.imshow(pivot.values, aspect='auto', origin='lower')
            ax4.set_yticks(range(len(pivot.index)))
            ax4.set_yticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
            ax4.set_xticks(range(0,24,2))
            ax4.set_xticklabels([str(h) for h in range(0,24,2)])
            ax4.set_xlabel('Hour of day')
            ax4.set_ylabel('Weekday')
            fig4.colorbar(im, ax=ax4, label='Messages')
            st.pyplot(fig4, transparent=True)

    st.subheader("Response time analysis")
    resp, resp_summary = build_response_times(df_view)
    if not resp_summary:
        st.info("Not enough turn‑taking to compute response times.")
    else:
        rows = []
        for s, m in resp_summary.items():
            rows.append({
                'Sender': apply_alias(s),
                'Responses': m['count'],
                'Avg (min)': round(m['avg_sec']/60, 2),
                'Median (min)': round(m['median_sec']/60, 2),
                'P90 (min)': round(m['p90_sec']/60, 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        sel_sender = st.selectbox("Histogram for sender", [apply_alias(s) for s in resp.keys()])
        # map back to original key
        inv_alias = {apply_alias(k): k for k in resp.keys()}
        vals = np.array(resp[inv_alias[sel_sender]]) / 60  # minutes
        fig2, ax2 = plt.subplots()
        ax2.hist(vals, bins=30)
        ax2.set_xlabel('Reply time (minutes)')
        ax2.set_ylabel('Count')
        st.pyplot(fig2, transparent=True)

# ===== Media =====
with tab_media:
    st.subheader("Media & lengths")
    stats = get_high_level_stats(df_view)
    if not stats:
        st.info("No messages in range.")
    else:
        media = stats['media_counts']
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Photos", media['photos'])
        col_m2.metric("Videos", media['videos'])
        col_m3.metric("Shares/Links", media['shares/links'])
        col_m4.metric("Stickers", media['stickers'])

        avg_df = pd.DataFrame({
            'Sender': list(stats['avg_len_words'].keys()),
            'Avg words/msg': list(stats['avg_len_words'].values()),
            'Avg chars/msg': list(stats['avg_len_chars'].values()),
        })
        avg_df['Sender'] = avg_df['Sender'].map(apply_alias)
        st.dataframe(avg_df, use_container_width=True)

# ===== Search =====
with tab_search:
    st.subheader("Search messages")
    q = st.text_input("Text contains (case‑insensitive)", "")
    sender_filter = st.multiselect("Senders", sorted(df_view['sender'].unique()), default=[])
    tmp = df_view.copy()
    if sender_filter:
        tmp = tmp[tmp['sender'].isin(sender_filter)]
    if q:
        qlc = q.lower()
        tmp = tmp[tmp['text'].str.lower().str.contains(qlc, na=False)]
    show_cols = ['ts','sender','text','n_words','n_chars','has_photo','has_video','has_share','has_sticker']
    st.dataframe(tmp[show_cols], use_container_width=True)

# ===== Export =====
with tab_export:
    st.subheader("Export data")
    show_cols = ['ts','sender','text','n_words','n_chars','has_photo','has_video','has_share','has_sticker']
    csv = df_view[show_cols].to_csv(index=False).encode('utf-8')
    safe_title = re.sub(r'[^A-Za-z0-9_-]+','_', thread.title)[:40]
    st.download_button("Download CSV (messages)", data=csv, file_name=f"{safe_title}_messages.csv", mime="text/csv")

    # Word usage CSV
    dfw, _ = build_word_frequencies(df_view, top_n=200, stop_extra=extra_sw, bigrams=use_bigrams)
    if not dfw.empty:
        csv_words = dfw.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV (word usage)", data=csv_words, file_name=f"{safe_title}_word_usage.csv", mime="text/csv")

    # Analytics JSON summary
    stats = get_high_level_stats(df_view)
    if stats:
        summary = {
            'title': thread.title,
            'participants': [apply_alias(p) for p in thread.participants],
            'date_range': [str(start_date), str(end_date)],
            'stats': stats,
            'starters_by_day': starters_by_day(df_view),
            'longest_gap_hours': round(longest_gap(df_view)[0] / 3600, 2),
            'streak_days': day_streak(df_view),
        }
        json_bytes = json.dumps(summary, ensure_ascii=False, indent=2, default=str).encode('utf-8')

        st.download_button("Download JSON (summary)", data=json_bytes, file_name=f"{safe_title}_summary.json", mime="application/json")

st.caption("Made with ❤️ in Streamlit. Optional add‑ons: VADER/TextBlob sentiment, wordclouds, Indic tokenizers.")
