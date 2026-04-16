import streamlit as st
import torch
import requests
from transformers import pipeline
import plotly.graph_objects as go
import pandas as pd

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0f;
    color: #e8e4dc;
}
.stApp { background: #0d0d0f; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }

.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #161618; border-radius: 12px;
    padding: 6px; border: 1px solid #2a2a2e;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.05rem;
    letter-spacing: 1.5px; color: #888; border-radius: 8px;
    padding: 8px 28px; background: transparent; border: none; transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #f5c518 !important; background: #1f1f23 !important;
    border-bottom: 2px solid #f5c518 !important;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a1e 0%, #141416 100%);
    border: 1px solid #2a2a2e; border-radius: 14px;
    padding: 22px 28px; text-align: center; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #f5c518; }
.metric-label { font-size: 0.78rem; letter-spacing: 2px; text-transform: uppercase; color: #666; margin-bottom: 8px; }
.metric-value { font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem; color: #f5c518; line-height: 1; }
.metric-model { font-size: 0.8rem; color: #aaa; margin-top: 4px; }
.winner-badge {
    display: inline-block; background: #f5c518; color: #0d0d0f;
    font-family: 'Bebas Neue', sans-serif; letter-spacing: 1.5px;
    font-size: 0.75rem; padding: 3px 10px; border-radius: 20px;
    margin-left: 8px; vertical-align: middle;
}
.stTextArea textarea {
    background: #161618 !important; border: 1px solid #2a2a2e !important;
    border-radius: 10px !important; color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #f5c518 !important;
    box-shadow: 0 0 0 2px rgba(245,197,24,0.15) !important;
}
.stButton > button {
    background: #f5c518 !important; color: #0d0d0f !important;
    font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 2px !important;
    font-size: 1rem !important; border: none !important; border-radius: 8px !important;
    padding: 10px 32px !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.result-box { border-radius: 14px; padding: 28px 32px; margin-top: 16px; border: 1px solid; text-align: center; }
.result-positive { background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.35); }
.result-negative { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.35); }
.result-emoji { font-size: 3rem; }
.result-label { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; letter-spacing: 3px; margin-top: 8px; }
.result-confidence { font-size: 0.9rem; color: #888; margin-top: 4px; }
.imdb-card {
    background: #161618; border: 1px solid #2a2a2e; border-radius: 14px;
    padding: 20px 24px; display: flex; gap: 20px;
    align-items: flex-start; margin-bottom: 14px;
}
.imdb-rating {
    background: #f5c518; color: #0d0d0f; font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem; padding: 6px 12px; border-radius: 8px;
    min-width: 60px; text-align: center;
}
.imdb-title { font-size: 1.05rem; font-weight: 600; color: #e8e4dc; }
.imdb-year  { font-size: 0.82rem; color: #666; margin-top: 3px; }
.imdb-plot  { font-size: 0.88rem; color: #aaa; margin-top: 8px; line-height: 1.5; }
hr { border-color: #2a2a2e !important; }
.section-header {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem;
    letter-spacing: 3px; color: #f5c518;
    border-bottom: 1px solid #2a2a2e; padding-bottom: 8px; margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STATIC RESULTS  (your actual training metrics)
# ─────────────────────────────────────────────
MODEL_RESULTS = {
    "ELECTRA-small": {
        "accuracy":  0.8964,
        "f1":        0.8960,
        "precision": 0.8950,
        "recall":    0.8970,
        "eval_loss": 0.2780,
        "params_M":  13.5,
        "epochs": 3,
        "lr": "2e-5",
        "color": "#60a5fa",
    },
    "DeBERTa-base": {
        "accuracy":  0.9124,
        "f1":        0.9121,
        "precision": 0.9136,
        "recall":    0.9107,
        "eval_loss": 0.2473,
        "params_M":  139,
        "epochs": 3,
        "lr": "2e-5",
        "color": "#f5c518",
    },
    "DistilBERT-base": {
        "accuracy":  0.8928,
        "f1":        0.8969,
        "precision": 0.8659,
        "recall":    0.9302,
        "eval_loss": 0.3119,
        "params_M":  66,
        "epochs": 3,
        "lr": "2e-5",
        "color": "#a78bfa",
    },
}

WINNER = max(MODEL_RESULTS, key=lambda m: MODEL_RESULTS[m]["accuracy"])


# ─────────────────────────────────────────────
# SENTIMENT PIPELINE
# Uses distilbert-base-uncased-finetuned-sst-2-english
# — a production-ready IMDb sentiment model, works instantly,
#   no training or .pth files needed.
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sentiment model…")
def load_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=True,
        max_length=512,
    )



from spellchecker import SpellChecker

_spell = SpellChecker()

def correct_spelling(text: str) -> str:
    words = text.split()
    corrected = []
    for word in words:
        fixed = _spell.correction(word)
        corrected.append(fixed if fixed else word)
    return " ".join(corrected)

def predict_sentiment(text: str):
    clf = load_pipeline()
    clean = " ".join(text.strip().lower().split())
    clean = correct_spelling(clean)   # ← new line
    result = clf(clean)[0]
    
    raw_label = result["label"]   # "POSITIVE" or "NEGATIVE"
    conf      = result["score"]   # confidence in the predicted class
    
    # prob_pos / prob_neg are always the true probabilities
    prob_pos = conf if raw_label == "POSITIVE" else 1 - conf
    prob_neg = 1 - prob_pos

    # ── Neutral zone: model is uncertain (neither side dominates strongly) ──
    NEUTRAL_THRESHOLD = 0.60   # tweak this: lower = more neutrals, higher = fewer
    if conf < NEUTRAL_THRESHOLD:
        label = "Neutral"
    elif raw_label == "POSITIVE":
        label = "Positive"
    else:
        label = "Negative"

    return label, conf, prob_neg, prob_pos


# ─────────────────────────────────────────────
# OMDB API
# ─────────────────────────────────────────────
OMDB_API_KEY = "131ad5d"   # replace with your free key from omdbapi.com

def search_omdb(query: str):
    url = f"http://www.omdbapi.com/?s={requests.utils.quote(query)}&type=movie&apikey={OMDB_API_KEY}"
    try:
        r    = requests.get(url, timeout=6)
        data = r.json()
        if data.get("Response") == "True":
            return data.get("Search", [])
    except Exception:
        pass
    return []

def get_omdb_detail(imdb_id: str):
    url = f"http://www.omdbapi.com/?i={imdb_id}&plot=short&apikey={OMDB_API_KEY}"
    try:
        r = requests.get(url, timeout=6)
        return r.json()
    except Exception:
        return {}


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 18px 0;">
  <div style="font-family:'Bebas Neue',sans-serif; font-size:3.2rem; letter-spacing:6px; color:#f5c518;">
    🎬 SENTIMENT BENCHMARK
  </div>
  <div style="font-size:0.95rem; color:#666; margin-top:4px; letter-spacing:1px;">
    ELECTRA · DeBERTa · DistilBERT — trained on IMDb (10K samples)
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊  MODEL PERFORMANCE", "🎥  REVIEW ANALYZER"])


# ═══════════════════════════════════════════════════════
# TAB 1 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Accuracy cards
    cols = st.columns(3)
    for i, (mname, mdata) in enumerate(MODEL_RESULTS.items()):
        badge = "<span class='winner-badge'>BEST</span>" if mname == WINNER else ""
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Accuracy{badge}</div>
              <div class="metric-value">{mdata['accuracy']*100:.2f}%</div>
              <div class="metric-model">{mname}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Grouped bar chart
    st.markdown("<div class='section-header'>METRICS COMPARISON</div>", unsafe_allow_html=True)
    metrics_order = ["accuracy", "f1", "precision", "recall"]
    fig = go.Figure()
    for mname, mdata in MODEL_RESULTS.items():
        fig.add_trace(go.Bar(
            name=mname,
            x=[m.capitalize() for m in metrics_order],
            y=[mdata[m] for m in metrics_order],
            marker_color=mdata["color"],
            text=[f"{mdata[m]*100:.1f}%" for m in metrics_order],
            textposition="outside",
            textfont=dict(size=11, color="#e8e4dc"),
        ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
        font=dict(color="#e8e4dc", family="DM Sans"),
        legend=dict(bgcolor="#161618", bordercolor="#2a2a2e", borderwidth=1),
        xaxis=dict(gridcolor="#1e1e22"),
        yaxis=dict(gridcolor="#1e1e22", range=[0.82, 0.96], tickformat=".0%"),
        margin=dict(t=20, b=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.markdown("<div class='section-header'>METRICS HEATMAP</div>", unsafe_allow_html=True)
    hm_metrics = ["Accuracy", "F1", "Precision", "Recall"]
    hm_keys    = ["accuracy", "f1", "precision", "recall"]
    hm_models  = list(MODEL_RESULTS.keys())
    hm_values  = [[MODEL_RESULTS[m][k] * 100 for k in hm_keys] for m in hm_models]
    hm_text    = [[f"{v:.2f}%" for v in row] for row in hm_values]

    fig2 = go.Figure(go.Heatmap(
        z=hm_values,
        x=hm_metrics,
        y=hm_models,
        text=hm_text,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        colorscale=[
            [0.0, "#1a1a2e"],
            [0.4, "#16213e"],
            [0.7, "#b8860b"],
            [1.0, "#f5c518"],
        ],
        showscale=True,
        colorbar=dict(
            tickformat=".1f",
            ticksuffix="%",
            bgcolor="#161618",
            bordercolor="#2a2a2e",
            tickfont=dict(color="#aaa"),
        ),
        zmin=86, zmax=94,
    ))
    fig2.update_layout(
        plot_bgcolor="#0d0d0f", paper_bgcolor="#0d0d0f",
        font=dict(color="#e8e4dc", family="DM Sans"),
        xaxis=dict(side="top", tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=13)),
        margin=dict(t=40, b=20, l=10, r=10),
        height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Detailed table
    st.markdown("<div class='section-header'>DETAILED RESULTS TABLE</div>", unsafe_allow_html=True)
    rows = []
    for mname, mdata in MODEL_RESULTS.items():
        rows.append({
            "Model":      mname,
            "Accuracy":   f"{mdata['accuracy']*100:.2f}%",
            "F1":         f"{mdata['f1']*100:.2f}%",
            "Precision":  f"{mdata['precision']*100:.2f}%",
            "Recall":     f"{mdata['recall']*100:.2f}%",
            "Eval Loss":  f"{mdata['eval_loss']:.4f}",
            "Params (M)": mdata["params_M"],
            "Epochs":     mdata["epochs"],
            "LR":         mdata["lr"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df.set_index("Model"), use_container_width=True)

    # Training config
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>TRAINING CONFIGURATION</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip([c1, c2, c3, c4], [
        ("DATASET",    "IMDb (Hugging Face)"),
        ("TRAIN SIZE", "10,000 samples"),
        ("TEST SIZE",  "5,000 samples"),
        ("MAX LENGTH", "256 tokens"),
    ]):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div style="font-size:1.1rem;font-weight:600;color:#e8e4dc;margin-top:4px">{val}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# TAB 2 — REVIEW ANALYZER (FIXED VERSION)
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # -------------------------------
    # MOVIE DATABASE (STATIC, RELIABLE)
    # -------------------------------
    MOVIES = {
    "Action": {
        "English": ["Inception", "Mad Max: Fury Road", "John Wick", "The Dark Knight", "Mission: Impossible – Fallout", "Die Hard", "Top Gun: Maverick", "Black Panther", "Edge of Tomorrow", "The Raid 2"],
        "Hindi": ["War", "Pathaan", "Dhoom 3", "Tiger Zinda Hai", "Baahubali: The Beginning", "KGF: Chapter 1", "Uri: The Surgical Strike", "Don", "Sholay", "Agneepath"],
        "Korean": ["The Roundup", "A Bittersweet Life", "The Man from Nowhere", "Train to Busan", "Veteran"],
        "Spanish": ["La Casa de Papel", "El apagón", "El desconocido"],
        "Japanese": ["Rurouni Kenshin", "13 Assassins", "Blade of the Immortal"]
    },
    "Drama": {
        "English": ["The Shawshank Redemption", "Forrest Gump", "Fight Club", "Schindler's List", "The Godfather", "12 Angry Men", "American Beauty", "Parasite", "The Pianist", "Whiplash"],
        "Hindi": ["Taare Zameen Par", "3 Idiots", "Gully Boy", "Mughal-E-Azam", "Dil Chahta Hai", "Swades", "Gangs of Wasseypur", "Masaan", "Udaan", "Kapoor & Sons"],
        "Korean": ["Burning", "Poetry", "Oasis", "The Wailing", "A Taxi Driver"],
        "French": ["Amélie", "La Vie en Rose", "Blue is the Warmest Colour", "The Class", "Mustang"],
        "Italian": ["Cinema Paradiso", "Life is Beautiful", "Bicycle Thieves", "The Great Beauty", "I Am Love"],
        "Tamil": ["Vikram Vedha", "Visaranai", "Pariyerum Perumal", "Super Deluxe", "Kaithi"]
    },
    "Sci-Fi": {
        "English": ["Interstellar", "The Matrix", "Arrival", "Blade Runner 2049", "Dune", "2001: A Space Odyssey", "Eternal Sunshine of the Spotless Mind", "Ex Machina", "Contact", "Coherence"],
        "Hindi": ["Ra.One", "Koi Mil Gaya", "PK", "Robot (Enthiran)", "Brahmastra", "Mr. India"],
        "Korean": ["Space Sweepers", "Snowpiercer", "The Host"],
        "Japanese": ["Ghost in the Shell", "Akira", "Nausicaä of the Valley of the Wind", "Paprika", "The End of Evangelion"]
    },
    "Thriller": {
        "English": ["Se7en", "Prisoners", "Gone Girl", "Zodiac", "No Country for Old Men", "Memento", "The Silence of the Lambs", "Knives Out", "Nightcrawler", "Oldboy (2013)"],
        "Hindi": ["Andhadhun", "Kahaani", "Drishyam", "A Wednesday", "Special 26", "Talaash", "Ek Hasina Thi"],
        "Korean": ["Oldboy", "I Saw the Devil", "Mother", "Memories of Murder", "The Yellow Sea"],
        "Spanish": ["The Invisible Guest", "Sleep Tight", "Cell 211", "The Body", "During the Storm"],
        "French": ["Cache", "Tell No One", "With a Friend Like Harry", "Inside", "À bout portant"]
    },
    "Comedy": {
        "English": ["The Grand Budapest Hotel", "Superbad", "In Bruges", "The Nice Guys", "What We Do in the Shadows", "Bridesmaids", "Airplane!", "Groundhog Day", "Some Like It Hot", "The Death of Stalin"],
        "Hindi": ["Hera Pheri", "Jaane Bhi Do Yaaro", "Andaz Apna Apna", "Golmaal", "Dhamaal", "Chupke Chupke", "Padosan", "Band Baaja Baaraat"],
        "Korean": ["Extreme Job", "Sunny", "Welcome to Dongmakgol"],
        "Japanese": ["Tampopo", "The Happiness of the Katakuris", "Welcome Back, Mr. McDonald"],
        "Spanish": ["El otro lado de la cama", "La comunidad", "Ocho apellidos vascos"]
    },
    "Horror": {
        "English": ["Hereditary", "The Shining", "Get Out", "Midsommar", "A Quiet Place", "The Witch", "It Follows", "Suspiria", "The Thing", "Mandy"],
        "Hindi": ["Stree", "Tumbbad", "Bhool Bhulaiyaa", "Raat", "Pari", "1920", "Darr"],
        "Korean": ["A Tale of Two Sisters", "The Wailing", "Gonjiam: Haunted Asylum", "I Saw the Devil", "The Host"],
        "Japanese": ["Ringu", "Ju-On: The Grudge", "Audition", "Onibaba", "House (Hausu)"],
        "Spanish": ["REC", "The Orphanage", "El espinazo del diablo", "Veronica", "No matarás"]
    },
    "Romance": {
        "English": ["Before Sunrise", "Eternal Sunshine of the Spotless Mind", "Her", "Carol", "Call Me by Your Name", "In the Mood for Love", "Brokeback Mountain", "Portrait of a Lady on Fire", "The Notebook", "About Time"],
        "Hindi": ["Dilwale Dulhania Le Jayenge", "Kal Ho Naa Ho", "Veer-Zaara", "Jab We Met", "Rockstar", "Lootera", "Dil Dhadakne Do", "Kapoor & Sons"],
        "Korean": ["My Sassy Girl", "A Moment to Remember", "Architecture 101", "Our Little Sister", "Be with You"],
        "French": ["Amélie", "Portrait of a Lady on Fire", "Blue is the Warmest Colour", "The Intouchables"],
        "Tamil": ["Alaipayuthey", "Roja", "Bombay", "Minnale", "96"]
    },
    "Animation": {
        "English": ["Spirited Away", "WALL-E", "Spider-Man: Into the Spider-Verse", "The Iron Giant", "Persepolis", "The Incredibles", "Fantasia", "Grave of the Fireflies", "Loving Vincent"],
        "Japanese": ["My Neighbor Totoro", "Howl's Moving Castle", "Nausicaä of the Valley of the Wind", "The Tale of Princess Kaguya", "Wolf Children", "A Silent Voice", "Your Name", "Millennium Actress"],
        "French": ["The Triplets of Belleville", "Ernest & Celestine", "A Monster in Paris", "The Rabbi's Cat"]
    },
    "Documentary": {
        "English": ["13th", "Won't You Be My Neighbor?", "Searching for Sugar Man", "Amy", "The Act of Killing", "Icarus", "Bowling for Columbine", "Blackfish", "Free Solo", "The Cave"],
        "Hindi": ["An Insignificant Man", "Katiyabaaz", "Budhia Singh: Born to Run"],
        "French": ["Man on Wire", "Être et avoir"]
    },
    "Crime": {
        "English": ["The Godfather", "Goodfellas", "Pulp Fiction", "Heat", "L.A. Confidential", "Chinatown", "City of God", "Sicario", "Prisoners", "Zodiac"],
        "Hindi": ["Gangs of Wasseypur", "Satya", "Company", "Black Friday", "D-Day", "Shootout at Lokhandwala", "Once Upon a Time in Mumbaai"],
        "Korean": ["New World", "A Violent Prosecutor", "The Gangster, the Cop, the Devil"],
        "Spanish": ["Caníbal", "La zona", "El Incidente"],
        "Italian": ["Il Divo", "Gomorrah", "The Consequences of Love"]
    }
}

    left, right = st.columns([1.1, 1], gap="large")

    # =========================
    # LEFT SIDE — ANALYZER
    # =========================
    with left:
        st.markdown("<div class='section-header'>ANALYZE A REVIEW</div>", unsafe_allow_html=True)

        # Step 1: Select Genre
        genre = st.selectbox("Select Genre", list(MOVIES.keys()))

        # Step 2: Select Language
        language = st.selectbox("Select Language", list(MOVIES[genre].keys()))

        # Step 3: Select Movie
        movie = st.selectbox("Select Movie", MOVIES[genre][language])

        st.markdown(f"""
        <div style="margin-top:10px; font-size:0.9rem; color:#888;">
        🎬 Selected: <b>{movie}</b> ({language}, {genre})
        </div>
        """, unsafe_allow_html=True)

        # Review Input
        review_text = st.text_area(
            "Write your review",
            height=180,
            placeholder=f"Write your thoughts about {movie}..."
        )

        analyze_clicked = st.button("⚡ ANALYZE SENTIMENT", use_container_width=True)

        if analyze_clicked:
            if not review_text.strip():
                st.warning("Please write a review first.")
            else:
                with st.spinner("Analyzing..."):
                    label, conf, prob_neg, prob_pos = predict_sentiment(review_text)

                css_class = "result-positive" if label == "Positive" else "result-negative"
                emoji     = "😊" if label == "Positive" else "😞"
                color     = "#22c55e" if label == "Positive" else "#ef4444"

                st.markdown(f"""
                <div class="result-box {css_class}">
                  <div class="result-emoji">{emoji}</div>
                  <div class="result-label" style="color:{color}">{label}</div>
                  <div class="result-confidence">
                    Confidence: {conf*100:.1f}%  ·  Movie: {movie}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability chart
                fig = go.Figure(go.Bar(
                    x=[prob_pos, prob_neg],
                    y=["Positive", "Negative"],
                    orientation="h",
                    marker_color=["#22c55e", "#ef4444"],
                    text=[f"{prob_pos*100:.1f}%", f"{prob_neg*100:.1f}%"],
                    textposition="inside",
                    textfont=dict(color="white", size=13),
                ))
                fig.update_layout(
                    plot_bgcolor="#0d0d0f",
                    paper_bgcolor="#0d0d0f",
                    font=dict(color="#e8e4dc"),
                    xaxis=dict(range=[0, 1], visible=False),
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=130,
                )
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # RIGHT SIDE — INFO PANEL
    # =========================
    with right:
        st.markdown("<div class='section-header'>ABOUT</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:0.9rem; color:#aaa; line-height:1.6;">
        This tool analyzes the sentiment of your movie review using a 
        <b>pretrained DistilBERT model</b> fine-tuned on IMDb dataset.

        <br><br>
        <b>How it works:</b><br>
        • Text is cleaned and tokenized<br>
        • Passed through transformer model<br>
        • Output → Positive / Negative<br>
        • Confidence score + probability shown

        <br><br>
        🎯 Try writing different types of reviews for <b>{movie}</b> 
        and observe how the sentiment changes!
        </div>
        """, unsafe_allow_html=True)
