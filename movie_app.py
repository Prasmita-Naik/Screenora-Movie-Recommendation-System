"""
Screenora - Movie / Show Recommender (Hotstar-style Streamlit app)
------------------------------------------------------------------
Run:  streamlit run movie_app.py

Folder must contain:  movie_app.py, titles_new.csv, credits_new.csv
Poster images come from the free TMDB API (see TMDB_API_KEY below).
"""

import ast
import os
import random
from concurrent.futures import ThreadPoolExecutor
from html import escape
from urllib.parse import quote

import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TITLES_CSV = os.path.join(BASE_DIR, "titles_new.csv")
CREDITS_CSV = os.path.join(BASE_DIR, "credits_new.csv")
EMB_CACHE = os.path.join(BASE_DIR, "embeddings.npy")   # saved after 1st run

# Free key: https://www.themoviedb.org/settings/api  (use the "API Key (v3 auth)")
TMDB_API_KEY = ""          # <- paste your key here (or use the sidebar / env var)
IMG_BASE = "https://image.tmdb.org/t/p/"

st.set_page_config(
    page_title="Screenora",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #14234a 0%, #0c111b 55%) fixed;
    color: #e8ecf4;
}
.block-container {padding-top: 1rem; padding-bottom: 4rem; max-width: 1500px;}

/* ---------- navbar ---------- */
.logo {
    font-size: 32px; font-weight: 800; letter-spacing: 1px; padding-top: 2px;
    background: linear-gradient(90deg,#4cc9f0,#4361ee,#b5179e);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
/* active nav button (type="primary") */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg,#4361ee,#b5179e) !important;
    border-color: transparent !important; color:#fff !important;
}

/* ---------- inputs ---------- */
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important; color:#fff !important;
}
input {color:#fff !important;}
label, .stSelectbox label, .stTextInput label {color:#aab4cc !important; font-weight:600;}

/* ---------- hero ---------- */
.hero {
    position:relative; height: 430px; border-radius: 22px; overflow:hidden;
    background-size: cover; background-position: center top;
    box-shadow: 0 20px 60px rgba(0,0,0,.55); margin-bottom: 10px;
}
.hero::before {
    content:""; position:absolute; inset:0;
    background: linear-gradient(90deg, rgba(8,12,22,.96) 0%, rgba(8,12,22,.75) 38%, rgba(8,12,22,.05) 100%),
                linear-gradient(0deg, rgba(12,17,27,1) 0%, rgba(12,17,27,0) 35%);
}
.hero-body {position:absolute; left:48px; bottom:48px; max-width:620px;}
.hero-tag {
    display:inline-block; font-size:12px; font-weight:700; letter-spacing:1.5px;
    background: linear-gradient(90deg,#4361ee,#b5179e); padding:5px 12px;
    border-radius:20px; margin-bottom:14px;
}
.hero-title {font-size:50px; font-weight:800; line-height:1.05; margin-bottom:10px;}
.hero-meta {color:#cfd8ee; font-size:15px; margin-bottom:12px;}
.hero-desc {color:#b9c3dc; font-size:15px; line-height:1.55;}

/* ---------- section titles ---------- */
.sec {font-size:24px; font-weight:700; margin: 30px 0 14px 4px;}
.sec small {font-size:13px; color:#7f8bab; font-weight:500; margin-left:10px;}

/* ---------- cards ---------- */
.card {
    position:relative; border-radius:14px; overflow:hidden; aspect-ratio: 2/3;
    background:#141b2d; transition: transform .25s ease, box-shadow .25s ease;
    box-shadow: 0 6px 18px rgba(0,0,0,.45);
}
.card:hover {transform: scale(1.045); box-shadow: 0 14px 34px rgba(67,97,238,.45);}
.card img {width:100%; height:100%; object-fit:cover; display:block;}
.card .badge {
    position:absolute; top:10px; left:10px; font-size:11px; font-weight:700;
    padding:3px 9px; border-radius:8px; background:rgba(0,0,0,.65);
    border:1px solid rgba(255,255,255,.25);
}
.card .rating {
    position:absolute; top:10px; right:10px; font-size:12px; font-weight:700;
    padding:3px 8px; border-radius:8px; background:rgba(255,193,7,.95); color:#111;
}
.card .overlay {
    position:absolute; left:0; right:0; bottom:0; padding:34px 12px 12px 12px;
    background: linear-gradient(0deg, rgba(0,0,0,.92), rgba(0,0,0,0));
}
.card .t {font-weight:700; font-size:14px; line-height:1.2; margin-bottom:3px;}
.card .m {font-size:12px; color:#c4cde4;}

/* ---------- buttons ---------- */
.stButton > button {
    width:100%; border-radius:10px; font-weight:700; color:#fff;
    background: rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14);
    transition: all .2s ease; margin-top:-4px;
}
.stButton > button:hover {
    background: linear-gradient(90deg,#4361ee,#b5179e); border-color:transparent; color:#fff;
}
.stButton > button:focus {color:#fff; border-color:#4cc9f0;}

/* ---------- details dialog ---------- */
div[role="dialog"] {background:#0f1626 !important; border-radius:20px !important;}
.d-back {
    height:230px; border-radius:16px; background-size:cover; background-position:center;
    position:relative; margin-bottom:14px;
}
.d-back::after {
    content:""; position:absolute; inset:0; border-radius:16px;
    background: linear-gradient(0deg, #0f1626 0%, rgba(15,22,38,.1) 70%);
}
.d-title {font-size:34px; font-weight:800; margin-bottom:6px;}
.pill {
    display:inline-block; padding:4px 12px; margin:0 6px 8px 0; border-radius:20px;
    background:rgba(76,201,240,.12); border:1px solid rgba(76,201,240,.4);
    font-size:12px; font-weight:600; color:#8fe3ff;
}
.d-desc {color:#c4cde4; line-height:1.65; font-size:15px; margin:8px 0 12px 0;}
.d-cast {color:#8d99b8; font-size:13px; line-height:1.6;}
.d-cast b {color:#dfe6f7;}
.empty {text-align:center; padding:70px 0; color:#8d99b8; font-size:18px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA + MODEL  (cached so they load only once)
# ----------------------------------------------------------------------------
def parse_list(value):
    """'['drama', 'crime']' -> ['Drama', 'Crime']"""
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
        return [str(i).strip().title() for i in items]
    except Exception:
        return []


@st.cache_resource(show_spinner="Loading your movie universe… (first run builds the AI index)")
def load_engine():
    titles = pd.read_csv(TITLES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    actors = (
        credits.groupby("id")["name"].agg(list).str.join(" , ").reset_index(name="actors")
    )
    movies = pd.merge(titles, actors, on="id", how="left")

    # text columns -> empty string (same idea as in your notebook)
    for col in ["description", "age_certification", "genres",
                "production_countries", "actors", "type", "title"]:
        movies[col] = movies[col].fillna("").astype(str)

    # numeric columns stay numeric so we can sort / filter
    for col in ["release_year", "runtime", "seasons", "imdb_score",
                "imdb_votes", "tmdb_popularity", "tmdb_score"]:
        movies[col] = pd.to_numeric(movies[col], errors="coerce")
    movies["imdb_id"] = movies["imdb_id"].fillna("").astype(str)

    movies["profile"] = (
        movies["title"] + " " + movies["type"] + " " + movies["description"] + " "
        + movies["age_certification"] + " " + movies["genres"] + " "
        + movies["production_countries"] + " " + movies["seasons"].fillna("").astype(str)
        + " " + movies["actors"]
    )
    movies["genre_list"] = movies["genres"].apply(parse_list)
    movies = movies.reset_index(drop=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(EMB_CACHE):
        emb = np.load(EMB_CACHE)
        if emb.shape[0] != len(movies):
            emb = None
    else:
        emb = None
    if emb is None:
        emb = model.encode(
            movies["profile"].tolist(), batch_size=32,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        np.save(EMB_CACHE, emb)

    emb = emb.astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return movies, emb, index, model


movies, emb, index, model = load_engine()


@st.cache_resource
def art_store():
    """Plain dict that remembers poster/backdrop URLs for the whole session."""
    return {}


# ----------------------------------------------------------------------------
# POSTERS (TMDB)
# ----------------------------------------------------------------------------
def get_api_key():
    key = TMDB_API_KEY or os.environ.get("TMDB_API_KEY", "")
    try:
        key = key or st.secrets.get("TMDB_API_KEY", "")
    except Exception:
        pass
    return key or st.session_state.get("tmdb_key", "")


def _fetch_art(imdb_id, key):
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/find/{imdb_id}",
            params={"api_key": key, "external_source": "imdb_id"},
            timeout=6,
        )
        if r.ok:
            data = r.json()
            res = (data.get("movie_results") or []) + (data.get("tv_results") or [])
            if res:
                p, b = res[0].get("poster_path"), res[0].get("backdrop_path")
                return (
                    f"{IMG_BASE}w500{p}" if p else None,
                    f"{IMG_BASE}w1280{b}" if b else None,
                )
    except Exception:
        pass
    return (None, None)


def load_art(df):
    """Fetch posters for every row in df (in parallel) and store them."""
    key, store = get_api_key(), art_store()
    todo = [i for i in df["imdb_id"].unique() if i and i not in store]
    if key and todo:
        with ThreadPoolExecutor(max_workers=12) as ex:
            for imdb_id, res in zip(todo, ex.map(lambda x: _fetch_art(x, key), todo)):
                store[imdb_id] = res


def placeholder(title):
    palettes = [("#1f4068", "#162447"), ("#4a1942", "#2d142c"), ("#0f3057", "#00587a"),
                ("#3a0ca3", "#4361ee"), ("#6a040f", "#370617"), ("#064e3b", "#022c22")]
    c1, c2 = palettes[sum(map(ord, title)) % len(palettes)]
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450'>"
        f"<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
        f"<stop offset='0' stop-color='{c1}'/><stop offset='1' stop-color='{c2}'/>"
        "</linearGradient></defs><rect width='300' height='450' fill='url(#g)'/>"
        "<text x='150' y='215' fill='white' font-family='Arial' font-size='24' "
        f"font-weight='bold' text-anchor='middle'>{escape(title[:20])}</text>"
        "<text x='150' y='260' fill='rgba(255,255,255,0.6)' font-family='Arial' "
        "font-size='40' text-anchor='middle'>🎬</text></svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg)


def poster_of(row):
    return art_store().get(row["imdb_id"], (None, None))[0] or placeholder(row["title"])


def backdrop_of(row):
    b = art_store().get(row["imdb_id"], (None, None))[1]
    return b or art_store().get(row["imdb_id"], (None, None))[0] or placeholder(row["title"])


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def kind(row):
    return "SERIES" if str(row["type"]).upper() == "SHOW" else "MOVIE"


def runtime_text(row):
    if kind(row) == "SERIES":
        s = row["seasons"]
        return f"{int(s)} Season{'s' if s != 1 else ''}" if pd.notna(s) else "Series"
    r = row["runtime"]
    if pd.isna(r):
        return "Movie"
    return f"{int(r) // 60}h {int(r) % 60}m" if r >= 60 else f"{int(r)}m"


def year_text(row):
    return str(int(row["release_year"])) if pd.notna(row["release_year"]) else ""


def rating_text(row):
    return f"{row['imdb_score']:.1f}" if pd.notna(row["imdb_score"]) else "NA"


def open_details(idx):
    st.session_state["open_id"] = idx


def toggle_list(idx):
    wl = st.session_state.setdefault("watchlist", [])
    wl.remove(idx) if idx in wl else wl.append(idx)


def semantic_search(query, k=80):
    """Title match first, then FAISS semantic matches (same model as notebook)."""
    q = query.strip().lower()
    exact = movies[movies["title"].str.lower().str.contains(q, regex=False)]
    exact_idx = (exact.sort_values("tmdb_popularity", ascending=False)
                 .index.tolist()[:k])
    q_emb = model.encode([query], convert_to_numpy=True,
                         normalize_embeddings=True).astype("float32")
    _, ids = index.search(q_emb, k)
    seen = set(exact_idx)
    sem = [int(i) for i in ids[0] if int(i) not in seen]
    return exact_idx + sem


def similar_to(idx, k=8):
    _, ids = index.search(emb[idx:idx + 1], k + 1)
    return [int(i) for i in ids[0] if int(i) != idx][:k]


def apply_filters(df, typ, genre):
    if typ == "Movies":
        df = df[df["type"].str.upper() == "MOVIE"]
    elif typ == "Shows":
        df = df[df["type"].str.upper() == "SHOW"]
    if genre != "All":
        df = df[df["genre_list"].apply(lambda g: genre in g)]
    return df


def card_html(row):
    return (
        "<div class='card'>"
        f"<img src=\"{poster_of(row)}\" loading='lazy'/>"
        f"<div class='badge'>{kind(row)}</div>"
        f"<div class='rating'>⭐ {rating_text(row)}</div>"
        "<div class='overlay'>"
        f"<div class='t'>{escape(row['title'])}</div>"
        f"<div class='m'>{year_text(row)} • {runtime_text(row)}</div>"
        "</div></div>"
    )


def render_grid(df, section, cols=6):
    if df.empty:
        st.markdown("<div class='empty'>😕 Nothing found. Try another search or filter.</div>",
                    unsafe_allow_html=True)
        return
    load_art(df)
    rows = list(df.iterrows())
    for start in range(0, len(rows), cols):
        columns = st.columns(cols, gap="medium")
        for col, (idx, row) in zip(columns, rows[start:start + cols]):
            with col:
                st.markdown(card_html(row), unsafe_allow_html=True)
                st.button("▶ Details", key=f"{section}_{idx}",
                          on_click=open_details, args=(idx,))
        st.write("")


def section_title(text, sub=""):
    st.markdown(f"<div class='sec'>{text}<small>{sub}</small></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DETAILS POP-UP
# ----------------------------------------------------------------------------
@st.dialog("🎬  Now Showing", width="large")
def show_details(idx):
    row = movies.loc[idx]
    sims = movies.loc[similar_to(idx, 8)]
    load_art(pd.concat([movies.loc[[idx]], sims]))

    st.markdown(f"<div class='d-back' style=\"background-image:url('{backdrop_of(row)}')\"></div>",
                unsafe_allow_html=True)

    left, right = st.columns([1, 2.3], gap="large")
    with left:
        st.markdown(f"<div class='card'><img src=\"{poster_of(row)}\"/></div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='d-title'>{escape(row['title'])}</div>", unsafe_allow_html=True)
        pills = [f"⭐ {rating_text(row)} IMDb", year_text(row), runtime_text(row), kind(row)]
        if row["age_certification"]:
            pills.append(row["age_certification"])
        pills += row["genre_list"]
        st.markdown("".join(f"<span class='pill'>{escape(p)}</span>" for p in pills if p),
                    unsafe_allow_html=True)
        desc = row["description"] or "No description available."
        st.markdown(f"<div class='d-desc'>{escape(desc)}</div>", unsafe_allow_html=True)
        if row["actors"]:
            cast = ", ".join(row["actors"].split(" , ")[:8])
            st.markdown(f"<div class='d-cast'><b>Cast:</b> {escape(cast)}</div>",
                        unsafe_allow_html=True)
        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            st.button("▶  Play", key=f"play_{idx}")
        with b2:
            in_list = idx in st.session_state.get("watchlist", [])
            st.button("✔ In My List" if in_list else "＋ Add to My List",
                      key=f"wl_{idx}", on_click=toggle_list, args=(idx,))

    st.markdown("<div class='sec' style='font-size:20px'>More Like This</div>",
                unsafe_allow_html=True)
    rows = list(sims.iterrows())
    for start in range(0, len(rows), 4):
        columns = st.columns(4, gap="medium")
        for col, (sidx, srow) in zip(columns, rows[start:start + 4]):
            with col:
                st.markdown(card_html(srow), unsafe_allow_html=True)
                if st.button("▶ Open", key=f"sim_{idx}_{sidx}"):
                    st.session_state["open_id"] = sidx
                    st.rerun()


# ----------------------------------------------------------------------------
# NAVIGATION HELPERS
# ----------------------------------------------------------------------------
PAGES = ["Home", "Movies", "Shows", "My List"]
PAGE_LABELS = {"Home": "🏠 Home", "Movies": "🎥 Movies", "Shows": "📺 Shows", "My List": "📌 My List"}
SORT_COLS = {"Popularity": "tmdb_popularity", "IMDb Rating": "imdb_score", "Newest": "release_year"}


def go(page_name):
    """Navbar click: switch page, clear any search, reset 'load more'."""
    st.session_state["page"] = page_name
    st.session_state["search_q"] = ""
    st.session_state["limit"] = 12


def clear_search():
    """Back button: clear the search box and return to the page we were on."""
    st.session_state["search_q"] = ""


def load_more():
    st.session_state["limit"] = st.session_state.get("limit", 12) + 12


def browse(df, title, sort_by):
    """Full catalogue page with sorting + Load more."""
    df = df.sort_values(SORT_COLS[sort_by], ascending=False, na_position="last")
    limit = st.session_state.get("limit", 12)
    section_title(title, f"{len(df)} titles")
    render_grid(df.head(limit), "browse")
    if limit < len(df):
        _, mid, _ = st.columns([2, 1, 2])
        with mid:
            st.button("⬇  Load more", key="more_btn", on_click=load_more)


# ----------------------------------------------------------------------------
# PAGE
# ----------------------------------------------------------------------------
st.session_state.setdefault("page", "Home")
st.session_state.setdefault("limit", 12)
page = st.session_state["page"]

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.text_input("TMDB API key (for posters)", type="password", key="tmdb_key",
                  help="Free from themoviedb.org → Settings → API")
    st.caption("Without a key, stylish placeholder posters are shown.")

# ---------------- NAVBAR (real, clickable buttons) ----------------
nav = st.columns([2.4, 1, 1, 1, 1.1, 1.6], gap="small")
with nav[0]:
    st.markdown("<div class='logo'>🎬 Screenora</div>", unsafe_allow_html=True)
for col, name in zip(nav[1:5], PAGES):
    with col:
        st.button(PAGE_LABELS[name], key=f"nav_{name}",
                  type="primary" if page == name else "secondary",
                  on_click=go, args=(name,))
st.write("")

# ---------------- SEARCH + FILTERS ----------------
all_genres = sorted({g for gl in movies["genre_list"] for g in gl})
c1, c2, c3 = st.columns([4, 1.4, 1.5])
with c1:
    query = st.text_input(
        "Search", key="search_q",
        placeholder="🔍  Search movies, shows, actors or a mood like 'space adventure'…",
        label_visibility="collapsed")
with c2:
    sort_by = "Popularity"
    if page in ("Movies", "Shows"):
        typ = page
        sort_by = st.selectbox("Sort", list(SORT_COLS), key="sort_by",
                               label_visibility="collapsed")
    elif page == "Home":
        typ = st.selectbox("Type", ["All", "Movies", "Shows"], label_visibility="collapsed")
    else:
        typ = "All"
with c3:
    genre = st.selectbox("Genre", ["All"] + all_genres, label_visibility="collapsed")

# Open details pop-up if a card was clicked
open_id = st.session_state.pop("open_id", None)
if open_id is not None:
    show_details(int(open_id))

# ============================================================================
# VIEW 1: SEARCH RESULTS  (with Back button)
# ============================================================================
if query.strip():
    st.button("←  Back", key="back_btn", on_click=clear_search)
    ids = semantic_search(query)
    res = apply_filters(movies.loc[ids], typ, genre).head(24)
    section_title(f"Results for “{escape(query)}”", f"{len(res)} titles")
    render_grid(res, "search")

# ============================================================================
# VIEW 2: MOVIES / SHOWS PAGES
# ============================================================================
elif page in ("Movies", "Shows"):
    pool = apply_filters(movies, page, genre)
    icon = "🎥" if page == "Movies" else "📺"
    if pool.empty:
        st.markdown("<div class='empty'>😕 No titles for this filter.</div>", unsafe_allow_html=True)
    else:
        browse(pool, f"{icon} All {page}", sort_by)

# ============================================================================
# VIEW 3: MY LIST PAGE
# ============================================================================
elif page == "My List":
    wl = st.session_state.get("watchlist", [])
    if not wl:
        st.markdown(
            "<div class='empty'>📌 Your list is empty.<br><br>"
            "<small>Open any title and tap <b>＋ Add to My List</b>.</small></div>",
            unsafe_allow_html=True)
    else:
        mine = apply_filters(movies.loc[wl[::-1]], "All", genre)   # newest added first
        section_title("📌 My List", f"{len(mine)} titles")
        render_grid(mine, "mylist_page")

# ============================================================================
# VIEW 4: HOME
# ============================================================================
else:
    pool = apply_filters(movies, typ, genre)

    if pool.empty:
        st.markdown("<div class='empty'>😕 No titles for this filter.</div>", unsafe_allow_html=True)
    else:
        popular = pool.sort_values("tmdb_popularity", ascending=False)

        # Hero banner (random from top 10, stays stable while you browse)
        top10 = popular.head(10)
        hero_key = f"hero_{typ}_{genre}"
        if hero_key not in st.session_state:
            st.session_state[hero_key] = int(random.choice(top10.index.tolist()))
        hidx = st.session_state[hero_key]
        hero = movies.loc[hidx]
        load_art(movies.loc[[hidx]])
        desc = hero["description"]
        desc = (desc[:230] + "…") if len(desc) > 230 else desc
        st.markdown(
            f"""<div class="hero" style="background-image:url('{backdrop_of(hero)}')">
                <div class="hero-body">
                  <div class="hero-tag">🔥 TRENDING NOW</div>
                  <div class="hero-title">{escape(hero['title'])}</div>
                  <div class="hero-meta">⭐ {rating_text(hero)} &nbsp;•&nbsp; {year_text(hero)}
                     &nbsp;•&nbsp; {runtime_text(hero)} &nbsp;•&nbsp; {escape(', '.join(hero['genre_list'][:3]))}</div>
                  <div class="hero-desc">{escape(desc)}</div>
                </div></div>""",
            unsafe_allow_html=True,
        )
        hb1, _ = st.columns([1, 4])
        with hb1:
            st.button("ℹ️  More Info", key="hero_btn", on_click=open_details, args=(hidx,))

        # My List preview row
        wl = st.session_state.get("watchlist", [])
        if wl:
            section_title("📌 My List")
            render_grid(movies.loc[wl[::-1][:6]], "mylist")

        section_title("🔥 Trending Now", "most popular right now")
        render_grid(popular.head(6), "trend")

        top_rated = pool[pool["imdb_votes"].fillna(0) >= 10000].sort_values(
            "imdb_score", ascending=False)
        section_title("🏆 Top Rated", "IMDb 10k+ votes")
        render_grid(top_rated.head(6), "top")

        if typ in ("All", "Movies"):
            m = pool[pool["type"].str.upper() == "MOVIE"].sort_values(
                "tmdb_popularity", ascending=False)
            section_title("🎥 Popular Movies")
            render_grid(m.head(6), "movies")

        if typ in ("All", "Shows"):
            s = pool[pool["type"].str.upper() == "SHOW"].sort_values(
                "tmdb_popularity", ascending=False)
            section_title("📺 Binge-worthy Shows")
            render_grid(s.head(6), "shows")

        newest = pool.sort_values("release_year", ascending=False).head(6)
        section_title("🆕 Recently Released")
        render_grid(newest, "new")
