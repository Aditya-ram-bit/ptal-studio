os, sys, tempfile, requests, re

# ── Paths — relative so works locally AND on Hugging Face ──────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR  = os.path.join(BASE_DIR, "core_engine")
STAGING_DIR = os.path.join(BASE_DIR, "web_staging")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_frames")

if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

import config, grid_gen, engine, visualizer
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PTAL Studio", page_icon="🗺️",
    layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.stApp { background: #12101A !important; color: #D4CCF0 !important; }
[data-testid="stSidebar"] { background: #1A1726 !important; border-right: 1px solid #2A2540 !important; }
@media (prefers-color-scheme: light) {
    .stApp { background: #F7F4FD !important; color: #3A3060 !important; }
    [data-testid="stSidebar"] { background: #EEE9FA !important; border-right: 1px solid #DDD6F5 !important; }
    h1 { color: #5A4E96 !important; }
    h2,h3 { color: #7060A8 !important; }
    .stCaption, small { color: #9080C0 !important; }
    hr { border-color: #DDD6F5 !important; }
    [data-testid="stFileUploadDropzone"] { background: #F0EBF9 !important; border-color: #C8BEE8 !important; }
    [data-testid="stFileUploadDropzone"] * { color: #7868AA !important; }
    .stButton > button { color: #3A3060 !important; }
}
h1 { font-family:'Inter',sans-serif !important; font-weight:600 !important; font-size:24px !important; color:#C4B6F0 !important; letter-spacing:-0.4px; }
h2,h3 { font-family:'Inter',sans-serif !important; font-weight:500 !important; font-size:15px !important; color:#A090D8 !important; }
.stCaption, small { font-size:12px !important; color:#6E5FA8 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { font-size:13px !important; font-weight:400 !important; color:#B8AEDC !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-family:'Inter',sans-serif !important; font-weight:500 !important; font-size:13.5px !important; color:#C8BFEA !important; letter-spacing:0.4px; text-transform:uppercase; }
.stButton > button { width:100%; background:linear-gradient(160deg,#3B2E68 0%,#4E3E80 100%) !important; color:#DDD4FF !important; border:1px solid #5A4A90 !important; border-radius:10px !important; padding:11px 20px !important; font-family:'Inter',sans-serif !important; font-weight:500 !important; font-size:13px !important; letter-spacing:0.3px; transition:all 0.18s ease !important; box-shadow:0 2px 10px rgba(100,80,180,0.3) !important; }
.stButton > button:hover { background:linear-gradient(160deg,#4A3A78 0%,#5E4E92 100%) !important; box-shadow:0 4px 16px rgba(120,100,200,0.45) !important; transform:translateY(-1px); }
[data-testid="stFileUploadDropzone"] { background:#1E1A2E !important; border:1.5px dashed #3E3460 !important; border-radius:12px !important; }
[data-testid="stFileUploadDropzone"] * { color:#7868A8 !important; font-size:12.5px !important; }
[data-baseweb="slider"] [role="slider"] { background:#8070C0 !important; border-color:#9080D0 !important; }
[data-testid="stRadio"] label { font-size:13px !important; color:#B0A0D8 !important; }
[data-testid="stAlert"] { border-radius:10px !important; font-size:13px !important; border:none !important; }
hr { border-color:#2A2440 !important; margin:14px 0 !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#3E3460; border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:#504080; }
/* URL input box */
[data-testid="stTextInput"] input {
    background:#1E1A2E !important;
    border:1px solid #3E3460 !important;
    border-radius:8px !important;
    color:#C4B6F0 !important;
    font-size:12.5px !important;
    padding:8px 12px !important;
}
</style>
""", unsafe_allow_html=True)

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)


# ── URL download helper ─────────────────────────────────────────────────────
def _gdrive_direct(url: str) -> str:
    """Convert Google Drive share URL to direct download URL."""
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return url

def _dropbox_direct(url: str) -> str:
    """Convert Dropbox share URL to direct download."""
    return url.replace("?dl=0", "?dl=1").replace("www.dropbox.com", "dl.dropboxusercontent.com")

def download_gtfs_from_url(url: str, staging_dir: str) -> str | None:
    """
    Downloads a GTFS zip from a URL into staging_dir.
    Supports Google Drive, Dropbox, and direct links.
    Returns local file path or None on failure.
    """
    url = url.strip()
    if not url:
        return None

    if "drive.google.com" in url:
        url = _gdrive_direct(url)
    elif "dropbox.com" in url:
        url = _dropbox_direct(url)

    try:
        resp = requests.get(url, stream=True, timeout=120,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        # Detect filename from headers or URL
        cd = resp.headers.get("content-disposition", "")
        fname_match = re.search(r'filename="?([^";\n]+)"?', cd)
        if fname_match:
            fname = fname_match.group(1).strip()
        else:
            # Use a hash of the URL so multiple URLs never collide on disk
            import hashlib
            fname = "gtfs_" + hashlib.md5(url.encode()).hexdigest()[:8] + ".zip"
        if not fname.endswith(".zip"):
            fname = fname + ".zip"

        out_path = os.path.join(staging_dir, fname)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                f.write(chunk)
        return out_path

    except Exception as e:
        st.error(f"Download failed for `{url[:60]}…`: {e}")
        return None


# ── Platform detection ──────────────────────────────────────────────────────
_ON_HF    = bool(os.environ.get("SPACE_ID"))
_ON_ST    = bool(os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("IS_STREAMLIT_CLOUD"))
_ON_LOCAL = not _ON_HF and not _ON_ST

_UPLOAD_LIMIT_MB = 16 if _ON_HF else (200 if _ON_ST else None)
_UPLOAD_LIMIT_B  = _UPLOAD_LIMIT_MB * 1024 * 1024 if _UPLOAD_LIMIT_MB else None

def _platform_hint():
    if _ON_HF:
        return "🤗 **Hugging Face** — 16 MB file limit. Use URL input for larger files."
    if _ON_ST:
        return "☁️ **Streamlit Cloud** — ~200 MB file limit. Use URL input for larger files."
    return "💻 **Local** — no upload size limit."


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Parameters")
    st.caption("Set thresholds before running your analysis.")
    st.markdown("")

    framework = st.radio(
        "Evaluation framework",
        ["Delhi NCT", "TfL London"],
        index=0 if getattr(config, "PTAL_BAND_SYSTEM", "delhi") == "delhi" else 1
    )

    ddr_value = st.slider(
        "Walking decay ratio (DDR)",
        min_value=0.5, max_value=2.0,
        value=float(getattr(config, "DEFAULT_DDR", 1.3)), step=0.05
    )

    st.markdown("---")
    st.markdown("### Upload GTFS")

    # Platform hint — tells user exactly what they can do here
    st.info(_platform_hint())

    # Default to URL on HF since file upload is nearly useless there
    input_method = st.radio(
        "Input method",
        ["📁 File upload", "🔗 URL / Drive link"],
        index=1 if _ON_HF else 0,
        label_visibility="collapsed"
    )

    uploaded_files = []
    url_inputs     = []

    if input_method == "📁 File upload":
        uploaded_files = st.file_uploader(
            "Drop .zip archives here",
            type=["zip"], accept_multiple_files=True
        )
        # Per-file size warning
        if uploaded_files and _UPLOAD_LIMIT_B:
            for uf in uploaded_files:
                if uf.size > _UPLOAD_LIMIT_B:
                    st.warning(
                        f"**{uf.name}** ({uf.size//(1024*1024)} MB) exceeds the "
                        f"{_UPLOAD_LIMIT_MB} MB limit — switch to URL input for this file."
                    )
    else:
        st.caption("One URL per line — Google Drive, Dropbox, or direct .zip link.")
        raw_urls = st.text_area(
            "GTFS URLs",
            placeholder="https://drive.google.com/file/d/...\nhttps://example.com/gtfs.zip",
            height=110,
            label_visibility="collapsed"
        )
        url_inputs = [u.strip() for u in raw_urls.splitlines() if u.strip()]
        if url_inputs:
            st.caption(f"{len(url_inputs)} URL(s) queued.")

    st.markdown("---")
    st.markdown("### GeoJSON Overlay")
    st.caption("Optional — overlay boundaries, zones, or POI layers on the map.")
    geojson_files = st.file_uploader(
        "Drop .geojson or .json files",
        type=["geojson", "json"],
        accept_multiple_files=True,
        key="geojson_upload"
    )
    raw_gj_urls = st.text_area(
        "Or paste GeoJSON URLs (one per line)",
        placeholder="https://example.com/boundary.geojson",
        height=80,
        key="geojson_urls",
        label_visibility="visible"
    )

    st.markdown("")
    trigger_compile = st.button("✦  Compile PTAL Map")


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("PTAL Analysis Studio")
st.caption("Upload GTFS archives or paste links · configure parameters · generate an accessibility score map.")
st.markdown("---")

if trigger_compile:
    has_files = bool(uploaded_files)
    has_urls  = bool(url_inputs)

    has_geojson_files = bool(geojson_files)
    has_geojson_urls  = [u.strip() for u in raw_gj_urls.splitlines() if u.strip()] if raw_gj_urls else []
    has_any_geojson   = has_geojson_files or bool(has_geojson_urls)

    if not has_files and not has_urls and not has_any_geojson:
        st.error("Provide at least one GTFS .zip or GeoJSON file/URL.")
    else:
        try:
            saved = []

            with st.spinner("Preparing GTFS data…"):
                # File uploads
                if has_files:
                    for uf in uploaded_files:
                        p = os.path.join(STAGING_DIR, uf.name)
                        with open(p, "wb") as f:
                            f.write(uf.getbuffer())
                        saved.append(p)

                # URL downloads
                if has_urls:
                    for url in url_inputs:
                        p = download_gtfs_from_url(url, STAGING_DIR)
                        if p:
                            saved.append(p)
                            st.caption(f"✓ Downloaded: `{os.path.basename(p)}`")

            if not saved and not has_any_geojson:
                st.error("No valid GTFS files could be loaded. Check your URLs or uploads.")
                st.stop()

            # Validate each zip before passing to engine
            valid_zips = []
            for zpath in saved:
                ok, err = engine._validate_zip(zpath)
                if ok:
                    valid_zips.append(zpath)
                else:
                    st.warning(f"⚠️ **{os.path.basename(zpath)}** skipped — {err}")

            # GeoJSON — file uploads
            geojson_saved = []
            if geojson_files:
                for gf in geojson_files:
                    gp = os.path.join(STAGING_DIR, gf.name)
                    with open(gp, "wb") as f:
                        f.write(gf.getbuffer())
                    geojson_saved.append(gp)

            # GeoJSON — URL downloads
            for gurl in has_geojson_urls:
                import hashlib
                gfname = "geojson_" + hashlib.md5(gurl.encode()).hexdigest()[:8] + ".geojson"
                gpath  = os.path.join(STAGING_DIR, gfname)
                try:
                    import requests as _req
                    r = _req.get(gurl.strip(), timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                    r.raise_for_status()
                    with open(gpath, "wb") as f:
                        f.write(r.content)
                    geojson_saved.append(gpath)
                    st.caption(f"✓ GeoJSON downloaded: `{gfname}`")
                except Exception as e:
                    st.warning(f"GeoJSON URL failed: {e}")

            # Must have either valid GTFS or GeoJSON — not necessarily both
            if not valid_zips and not geojson_saved:
                st.error("Nothing to render — no valid GTFS or GeoJSON data found.")
                st.stop()

            with st.spinner("Rendering map…"):
                out_html = os.path.join(OUTPUT_DIR, "ptal_canvas_output.html")
                config.PTAL_BAND_SYSTEM = "delhi" if "Delhi" in framework else "tfl"
                config.DEFAULT_DDR      = ddr_value

                visualizer.render_ptal_map(
                    zip_paths=valid_zips, output_path=out_html,
                    ddr=ddr_value,
                    geojson_paths=geojson_saved if geojson_saved else None
                )

            # Cleanup staging
            for p in saved + geojson_saved:
                if os.path.exists(p):
                    os.remove(p)

            if os.path.exists(out_html):
                st.success("Map compiled successfully.")
                with open(out_html, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=750, scrolling=True)
            else:
                st.error("Map file was not created. Check engine logs.")

        except Exception as e:
            st.error(f"Engine error: {e}")

else:
    st.info("Choose an input method in the sidebar, load your GTFS data, then click **✦ Compile PTAL Map**.")
