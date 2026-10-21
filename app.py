"""SocialShield: ML-based Fake Account and Social Engineering Attack Detection.

Run:  streamlit run app.py
Change colours:  python src/set_theme.py midnight | light | terminal | slate
"""
import hashlib
import html
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from detector import ATTACK_INFO, SAFE, MessageDetector  # noqa: E402
from spread import cascade_stats, reach_over_time, simulate_cascade  # noqa: E402
import themes  # noqa: E402
import incidents  # noqa: E402
import ai_layer  # noqa: E402

st.set_page_config(page_title="SocialShield", page_icon="🛡️", layout="wide")
T = themes.active()
COLORS = T["alerts"]
ICONS = {"Phishing": "🎣", "Money Scam": "💸", "Spam": "📢", "Misinformation": "📰",
         "Impersonation": "🎭", "Adult Scam": "🔞", SAFE: "✅"}
ATTACKER_NAMES = {"bot": "Spam bot", "crypto_scam": "Money scammer", "phishing": "Phisher",
                  "porn_scam": "Adult-content scammer", "impersonation": "Impersonator",
                  "hate_speech": "Misinformation spreader", "genuine": "Genuine user"}
ALERT_TO_ATTACKER = {"Phishing": "phishing", "Money Scam": "crypto_scam", "Spam": "bot",
                     "Misinformation": "hate_speech", "Impersonation": "impersonation", "Adult Scam": "porn_scam"}
NORMAL_NAMES = {"active_user": "Active user", "casual_user": "Casual user", "news_updates": "News page",
                "influencer": "Influencer", "business": "Business", "community": "Community", "hobbyist": "Hobbyist"}
MIMICABLE = {"account_age_days": "Account age", "followers_count": "Followers", "following_count": "Following",
             "avg_likes": "Likes per post", "avg_comments": "Comments per post", "avg_shares": "Shares per post",
             "avg_chars": "Post length", "media_ratio": "Photos / videos", "night_ratio": "Night-time posting",
             "hour_std": "Posting-hour variety", "gap_mean_h": "Gap between posts", "gap_std_h": "Gap variety",
             "repeat_ratio": "Copy-paste posts", "url_ratio": "Links in posts", "has_bio": "Has bio",
             "has_location": "Has location", "profile_picture": "Profile picture",
             "follower_following_ratio": "Follower ratio"}


def hexa(color, alpha):
    """Hex colour plus alpha, for tinted backgrounds."""
    return f"{color}{int(alpha * 255):02x}"


st.markdown(f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1250px;}}
h1 {{font-size: 2rem !important; margin-bottom: 0.2rem !important;}}
.intro {{color:{T['muted']}; font-size:1rem; margin-bottom:1.2rem; line-height:1.5;}}
.badge {{display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:0.8rem; color:#fff;}}
.msg {{background:{T['card']}; border:1px solid {T['border']}; border-radius:10px; padding:14px 16px; line-height:1.65; margin:8px 0 14px;}}
.msg mark {{background:{hexa(COLORS['Misinformation'], 0.55)}; color:inherit; border-radius:4px; padding:0 3px;}}
.banner {{border-radius:12px; padding:14px 18px; margin:4px 0 6px; display:flex; justify-content:space-between; align-items:center;}}
.banner .t {{font-size:1.15rem; font-weight:700; letter-spacing:0.02em;}}
.banner .r {{font-size:1.6rem; font-weight:700;}}
.banner .s {{font-size:0.8rem; opacity:0.8;}}
.card {{background:{T['card']}; border:1px solid {T['border']}; border-radius:12px; padding:14px 16px; height:100%; min-height:92px;}}
.card .n {{font-size:1.9rem; font-weight:700; line-height:1.2;}}
.card .l {{color:{T['muted']}; font-size:0.85rem;}}
.small {{color:{T['muted']}; font-size:0.85rem;}}
.why li {{margin-bottom:6px;}}
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{display:none !important;}}
.tip {{cursor:help; color:{T['muted']}; font-size:0.8rem; margin-left:4px; border-bottom:1px dotted {T['muted']};}}
.risk {{display:inline-block; font-size:0.72rem; font-weight:700; padding:1px 8px; border-radius:999px; margin-left:6px; letter-spacing:0.03em;}}
.step {{background:{T['card']}; border:1px solid {T['border']}; border-radius:14px; padding:18px 18px 8px; height:100%;}}
.step .num {{display:inline-block; width:30px; height:30px; border-radius:50%; background:{T['primary']}; color:#fff; text-align:center; line-height:30px; font-weight:700; margin-bottom:8px;}}
.step .h {{font-weight:700; font-size:1.08rem; margin-bottom:4px;}}
.step .d {{color:{T['muted']}; font-size:0.92rem; margin-bottom:10px;}}
.tour {{min-height:178px;}}
.hero {{font-size:1.15rem; color:{T['muted']}; max-width:820px; line-height:1.6; margin:4px 0 22px;}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data (loaded once, cached)
def ensure_models():
    """On a fresh machine (e.g. Streamlit Cloud) the models do not exist yet: train them once."""
    if (ROOT / "models/summary.json").exists():
        return
    with st.status("First start: training the models. This happens only once and takes about 2 minutes.", expanded=True) as box:
        for script in ["train_account_model.py", "train_post_model.py", "precompute.py"]:
            st.write(f"Running {script} ...")
            r = subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                st.error(r.stderr[-2500:])
                st.stop()
        box.update(label="Models ready", state="complete")


ensure_models()


@st.cache_resource
def detector():
    return MessageDetector(ROOT / "models/post_model.joblib")


@st.cache_resource
def account_model():
    return joblib.load(ROOT / "models/account_model.joblib")["model"]


@st.cache_data
def load():
    accounts = pd.read_csv(ROOT / "models/scored_accounts.csv")
    posts = pd.read_csv(ROOT / "models/scored_posts.csv", keep_default_na=False)
    feats = pd.read_csv(ROOT / "models/account_features.csv", index_col=0)
    summary = json.loads((ROOT / "models/summary.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "reports/account_results.json").read_text(encoding="utf-8"))
    return accounts, posts, feats, summary, results


det = detector()
accounts, posts, feats, summary, results = load()
acc = accounts.set_index("user_id")


def h10(text):
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()[:10]


@st.cache_data(ttl=15, show_spinner=False)
def ai_status():
    return ai_layer.status()


@st.cache_data(show_spinner=False, max_entries=1000)
def ai_check(text, model):
    return ai_layer.check(text, model)


def ai_on():
    ready, model = ai_status()
    return (ready and st.session_state.get("ai_on", True)), model


def second_opinion(text, v):
    """Run the optional AI layer on one message and merge its opinion. Returns (verdict, detected_by, ai_result)."""
    on, model = ai_on()
    if not on:
        return v, "Hybrid detector", None
    with st.spinner("🤖 Asking the local AI for a second opinion ..."):
        ai = ai_check(text, model)
    v2, by = ai_layer.combine(v, ai, SAFE)
    return replace(v2, info=ATTACK_INFO.get(v2.label, {})), by, ai


def scan(text):
    return second_opinion(text, det.analyze(text))


def log_alert(key, source, v, text, sender="", by="Hybrid detector"):
    if v.label != SAFE:
        incidents.add(key, source, v.label, v.risk, text, sender, by)


# ---------------------------------------------------------------- shared pieces
def intro(title, text):
    st.title(title)
    st.markdown(f"<div class='intro'>{text}</div>", unsafe_allow_html=True)


def badge(label):
    return f'<span class="badge" style="background:{COLORS.get(label, "#666")}">{ICONS.get(label, "")} {label}</span>'


RISK_TIP = "How confident SocialShield is that this is an attack: rule matches plus the ML model's probability, capped at 100%."


def risk_word(label, risk):
    if label == SAFE or risk < 0.5:
        return ("SOME RISK", "#c98500") if risk >= 0.3 else ("LOW RISK", "#3f8f4f")
    return ("HIGH RISK", "#d03b3b") if risk >= 0.75 else ("MEDIUM RISK", "#c98500")


def risk_pill(label, risk):
    word, c = risk_word(label, risk)
    return f"<span class='risk' style='background:{hexa(c, 0.18)}; color:{c};'>{word}</span>"


def tip(text):
    return f"<span class='tip' title='{html.escape(text, quote=True)}'>ⓘ</span>"


def card(col, number, label, color=None, help=None):
    style = f"border-top:3px solid {color};" if color else ""
    extra = tip(help) if help else ""
    col.markdown(f"<div class='card' style='{style}'><div class='n'>{number}</div><div class='l'>{label}{extra}</div></div>",
                 unsafe_allow_html=True)


CAUTION = "#c98500"


def banner(label, risk, sub=""):
    c = COLORS.get(label, "#666")
    title = f"{label} detected"
    if label == SAFE:
        title, c = ("No threat detected", COLORS[SAFE]) if risk < 0.3 else ("Unclear, be careful", CAUTION)
    st.markdown(
        f"<div class='banner' style='background:{hexa(c, 0.16)}; border-left:5px solid {c};'>"
        f"<div><div class='t'>{ICONS.get(label, '') if title != 'Unclear, be careful' else '⚠️'} {title.upper()}</div><div class='s'>{sub}</div></div>"
        f"<div style='text-align:right'><div class='r'>{risk:.0%}</div><div class='s'>{risk_word(label, risk)[0].lower()} {tip(RISK_TIP)}</div></div></div>",
        unsafe_allow_html=True)


def highlighted(text, verdict):
    safe = html.escape(str(text))
    for h in verdict.rule_hits:
        for m in sorted(set(h["matches"]), key=len, reverse=True):
            em = html.escape(m)
            if em.strip():
                safe = safe.replace(em, f"<mark>{em}</mark>")
    return f"<div class='msg'>{safe}</div>"


def sender_box(user_id):
    if user_id not in acc.index:
        return
    a = acc.loc[user_id]
    fake = bool(a["predicted_fake"])
    st.markdown(f"**Sent by @{html.escape(a['username'])}**  ·  "
                f"{'🚩 ' + ATTACKER_NAMES.get(a['predicted_attack'], '') if fake else '✅ Genuine account'}")
    c = st.columns(3)
    c[0].metric("Fake-account risk", f"{a['fake_prob']:.0%}")
    c[1].metric("Account age", f"{a['account_age_days']} days")
    c[2].metric("Followers", f"{a['followers_count']:,}")
    if fake and isinstance(a["reasons"], str) and a["reasons"]:
        st.markdown("<ul class='why'>" + "".join(f"<li>{html.escape(r)}</li>" for r in a["reasons"].split(" | ")) + "</ul>",
                    unsafe_allow_html=True)


def ai_box(ai):
    if ai is None:
        return
    if not ai.available:
        st.caption(f"🤖 AI second opinion unavailable ({ai.error or 'offline'}).")
        return
    if ai.injection:
        st.warning("⚠️ This message contains instructions aimed at an AI filter (prompt injection). "
                   "SocialShield treats that as a manipulation attempt and ignores the instructions.")
    head = f"Scam · {ai.category}" if ai.verdict == "scam" else "Looks safe"
    c = COLORS.get(ai.category, COLORS[SAFE]) if ai.verdict == "scam" else COLORS[SAFE]
    st.markdown(f"<div class='msg' style='border-left:4px solid {c};'>🤖 <b>AI second opinion</b> "
                f"<span class='small'>({html.escape(ai.model)}, runs locally)</span><br><b>{head}</b> "
                f"({ai.confidence:.0%} confident): {html.escape(ai.reason)}</div>", unsafe_allow_html=True)


def respond_panel(key):
    inc = next((i for i in incidents.all_incidents() if i["key"] == key), None)
    if inc is None:
        return
    st.markdown("**Respond**")
    if inc["status"] == incidents.OPEN:
        b = st.columns(3)
        for col, (act, label) in zip(b, [("block", "🚫 Block sender"), ("quarantine", "🗄️ Quarantine"), ("dismiss", "✔️ Not a threat")]):
            if col.button(label, key=f"act_{act}_{key}", use_container_width=True):
                incidents.act(inc["id"], act)
                st.toast(f"Incident #{inc['id']}: {incidents.ACTIONS[act]}", icon="🛡️")
                st.rerun()
    else:
        a, b = st.columns([3, 1])
        a.success(f"Incident #{inc['id']}: **{inc['status']}** at {inc['action_time']}")
        if b.button("Undo", key=f"undo_{key}", use_container_width=True):
            incidents.reopen(inc["id"])
            st.rerun()


def verdict_view(text, v, sender_id=None, by="Hybrid detector", ai=None, incident_key=None):
    sub = "Detected by: signature rules + ML model" if by == "Hybrid detector" else f"Detected by: {by}"
    banner(v.label, v.risk, sub)
    st.markdown(highlighted(text, v), unsafe_allow_html=True)
    ai_box(ai)
    if v.label == SAFE and v.rule_hits:
        st.markdown(f"<span class='small'>Some warning signs were found (highlighted): "
                    f"{html.escape('; '.join(h['why'].lower() for h in v.rule_hits))}. Not enough to raise an alert, "
                    f"but check before you act.</span>", unsafe_allow_html=True)
    elif v.label == SAFE:
        st.caption("No scam patterns found and the ML model rates this message as normal.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Why it was flagged**")
            items = [f"{h['why']}: <i>{html.escape(', '.join(sorted(set(h['matches'])))[:70])}</i>" for h in v.rule_hits]
            top = max(v.ml_probs.items(), key=lambda t: t[1])
            items.append(f"ML model on its own: <b>{top[0]}</b> ({top[1]:.0%})")
            if ai is not None and ai.available and ai.verdict == "scam":
                items.append(f"Local AI: <b>{ai.category}</b> ({ai.confidence:.0%}): {html.escape(ai.reason)}")
            st.markdown("<ul class='why'>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>", unsafe_allow_html=True)
        with right:
            st.markdown("**About this attack**")
            st.markdown(f"{v.info['what']}\n\n**Goal:** {v.info['goal']}\n\n**What to do:** {v.info['action']}")
    if incident_key and v.label != SAFE:
        respond_panel(incident_key)
    if sender_id is not None:
        st.divider()
        sender_box(sender_id)


def fig_style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=44, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=T["muted"], size=12),
                      title_font=dict(color=T["text"], size=15))
    fig.update_xaxes(gridcolor=T["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=T["grid"], zeroline=False)
    return fig


def alert_row(label, risk, who, text, key):
    with st.container(border=True):
        a, b = st.columns([3.4, 1], vertical_alignment="center")
        a.markdown(f"{badge(label)}{risk_pill(label, risk)} &nbsp;<span class='small'>{who}</span><br>"
                   f"<span class='small'>{html.escape(text[:105])}{'…' if len(text) > 105 else ''}</span>",
                   unsafe_allow_html=True)
        return b.button("View", key=key, use_container_width=True)


def incident_row(i, prefix):
    with st.container(border=True):
        a, b = st.columns([2.6, 2.4], vertical_alignment="center")
        status = "" if i["status"] == incidents.OPEN else f" · <b>{i['status']}</b>"
        ai_tag = " · 🤖" if i.get("detected_by", "").startswith("AI") else ""
        a.markdown(f"{badge(i['alert'])}{risk_pill(i['alert'], i['risk'])} <span class='small'>#{i['id']} · {i['source']} · "
                   f"{i['time'][11:16]}{(' · @' + html.escape(i['sender'])) if i['sender'] else ''}{ai_tag}{status}</span><br>"
                   f"<span class='small'>{html.escape(i['text'][:110])}{'…' if len(i['text']) > 110 else ''}</span>",
                   unsafe_allow_html=True)
        with b:
            if i["status"] == incidents.OPEN:
                c = st.columns(3)
                for col, (act, label, helptext) in zip(c, [("block", "Block", "Block the sender's account"),
                                                          ("quarantine", "Quarantine", "Hide this message from the recipient"),
                                                          ("dismiss", "Not a threat", "Mark as a false alarm")]):
                    if col.button(label, key=f"{prefix}_{act}_{i['id']}", help=helptext, use_container_width=True):
                        incidents.act(i["id"], act)
                        st.toast(f"Incident #{i['id']}: {incidents.ACTIONS[act]}", icon="🛡️")
                        st.rerun()
            elif st.button("Undo", key=f"{prefix}_undo_{i['id']}", use_container_width=True):
                incidents.reopen(i["id"])
                st.rerun()


# ================================================================ PAGES
def page_home():
    st.title("🛡️ SocialShield")
    st.markdown("<div class='hero'>Finds <b>fake accounts</b> and <b>social media scams</b> (phishing links, money requests, "
                "spam, misinformation and impersonation) and explains every decision in plain language.</div>",
                unsafe_allow_html=True)
    st.subheader("Start here: a 3-minute tour")
    steps = [("1", "Scan a message", "Type any suspicious message, or pick an example, and see if it is a scam and why.", P["scan"], "Open Message Scanner"),
             ("2", "Check the DM inbox", "Messages written by our team, scanned for phishing, money requests and impersonation.", P["dms"], "Open DM Inbox"),
             ("3", "Watch the live feed", "Posts stream in like a real platform. Threats pop up the moment they are caught.", P["feed"], "Open Live Feed")]
    cols = st.columns(3)
    for col, (n, h, d, page, cta) in zip(cols, steps):
        with col:
            st.markdown(f"<div class='step tour'><div class='num'>{n}</div><div class='h'>{h}</div><div class='d'>{d}</div></div>",
                        unsafe_allow_html=True)
            st.page_link(page, label=f"**{cta}**", icon="➡️", use_container_width=True)
    st.write("")
    st.subheader("How SocialShield decides")
    c = st.columns(3)
    for col, (icon, h, d) in zip(c, [
            ("📋", "Scam signatures", "20 known scam patterns, like urgent payment requests or 'verify your account' links. Works like antivirus signatures."),
            ("🧠", "Message AI model", "A machine learning model trained on 131,284 real and malicious posts to recognise scam wording."),
            ("👤", "Account checker", "A second model that spots fake and bot accounts from their profile and posting behaviour.")]):
        col.markdown(f"<div class='step'><div class='h'>{icon} {h}</div><div class='d'>{d}</div></div>", unsafe_allow_html=True)
    st.write("")
    st.subheader("What the alerts mean")
    meaning = {"Phishing": "Fake login pages or forms that steal passwords and OTPs",
               "Money Scam": "Requests for money or fake investment profits",
               "Spam": "Mass promotional messages and chain forwards",
               "Misinformation": "False claims written to be shared quickly",
               "Impersonation": "Pretending to be a brand, official or someone you know",
               "Adult Scam": "Sexual bait leading to scams or blackmail"}
    c = st.columns(2)
    for i, (k, d) in enumerate(meaning.items()):
        c[i % 2].markdown(f"{badge(k)} &nbsp;<span class='small'>{d}</span>", unsafe_allow_html=True)
    st.write("")
    st.caption("More to explore in the sidebar: Overview (all numbers), Account Inspector, Spread Network and Evasion Lab. "
               "Data: Kaggle social media dataset (synthetic). DMs written by our team; spread paths simulated.")


def page_overview():
    intro("🛡️ Threat overview", "SocialShield scanned every account and every post in the dataset. "
          "This page summarises who is attacking and how.")
    c = st.columns(4)
    card(c[0], f"{summary['accounts_total']:,}", "Accounts scanned")
    card(c[1], f"{summary['accounts_flagged_fake']:,}", f"Fake accounts found ({summary['accounts_flagged_fake'] / summary['accounts_total']:.0%})", "#e66767", "Accounts the account model rates as fake or bot, based on profile and posting behaviour.")
    card(c[2], f"{summary['posts_total']:,}", "Posts scanned")
    card(c[3], f"{summary['posts_flagged']:,}", "Threat alerts raised", T["primary"])

    st.subheader("🔔 Alert center")
    inc = incidents.all_incidents()
    open_inc = [i for i in inc if i["status"] == incidents.OPEN]
    cnt = lambda s_: sum(i["status"] == s_ for i in inc)  # noqa: E731
    k = st.columns(4)
    card(k[0], len(open_inc), "Open alerts, need action", "#e66767", "Alerts from the live feed, DM inbox and scanner that nobody has handled yet.")
    card(k[1], cnt(incidents.ACTIONS["block"]), "Senders blocked", COLORS["Phishing"])
    card(k[2], cnt(incidents.ACTIONS["quarantine"]), "Messages quarantined", COLORS["Money Scam"])
    card(k[3], cnt(incidents.ACTIONS["dismiss"]), "False alarms", COLORS[SAFE])
    st.write("")
    if open_inc:
        st.markdown("**Latest open alerts**: block the sender, quarantine the message, or mark a false alarm")
        for i in sorted(open_inc, key=lambda x: (x["risk"], x["id"]), reverse=True)[:5]:
            incident_row(i, "ov")
    else:
        st.success("No open alerts. Start the Live Feed or scan a message to generate some.")
    st.page_link(P["incidents"], label="Open the full Incident Log", icon="📒")

    st.subheader("Who is attacking (whole dataset)")
    by, pb = summary["accounts_by_alert"], summary["posts_by_alert"]
    order = [("Spam", "sending spam"), ("Money Scam", "asking for money"), ("Phishing", "sending phishing links"),
             ("Misinformation", "spreading misinformation"), ("Impersonation", "impersonating others"), ("Adult Scam", "running adult scams")]
    for row in (order[:3], order[3:]):
        cols = st.columns(3)
        for col, (k, lab) in zip(cols, row):
            card(col, f"{ICONS[k]} {by.get(k, 0):,}", f"accounts {lab} · {pb.get(k, 0):,} posts", COLORS[k])
        st.write("")

    left, right = st.columns(2)
    with left:
        s = pd.Series(pb).sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[COLORS[k] for k in s.index],
                               text=[f"{v:,}" for v in s.values], textposition="outside",
                               hovertemplate="%{y}: %{x:,} posts<extra></extra>"))
        fig.update_layout(title="Malicious posts by attack type")
        fig.update_xaxes(range=[0, s.max() * 1.2])
        st.plotly_chart(fig_style(fig), use_container_width=True)
    with right:
        sp = pd.DataFrame(summary["spread_by_type"]).sort_values("shares_per_100_followers")
        names = [ATTACKER_NAMES.get(t, NORMAL_NAMES.get(t, t)) for t in sp["true_type"]]
        fig = go.Figure(go.Bar(x=sp["shares_per_100_followers"], y=names, orientation="h",
                               marker_color=[COLORS["Misinformation"] if t == "hate_speech" else T["neutral_bar"] for t in sp["true_type"]],
                               text=[f"{v:.1f}" for v in sp["shares_per_100_followers"]], textposition="outside",
                               hovertemplate="%{y}: %{x:.1f} shares per 100 followers<extra></extra>"))
        fig.update_layout(title="Misinformation spreads far beyond its audience")
        fig.update_xaxes(range=[0, sp["shares_per_100_followers"].max() * 1.18], title="Shares per 100 followers")
        st.plotly_chart(fig_style(fig), use_container_width=True)

    with st.expander("Model accuracy on held-out test data"):
        perf = pd.DataFrame(results["binary"]).T.drop(columns="confusion_matrix").astype(float)
        perf.columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        st.dataframe(perf.style.format("{:.3f}"), use_container_width=True)
        st.caption("Near perfect because the dataset is synthetic. The DM Inbox and Evasion Lab show realistic performance.")


@st.cache_data
def feed_source():
    test_users = set(accounts.loc[accounts["in_test_set"], "user_id"])
    return posts[posts["user_id"].isin(test_users)].sample(frac=1, random_state=3).reset_index(drop=True)


def page_feed():
    intro("🚨 Live threat feed", "Simulates a platform's incoming posts. Press <b>Start</b>: every post is scanned as it "
          "arrives and threats appear on the left. Press <b>View</b> on any alert to see why it was flagged.")
    src = feed_source()
    ss = st.session_state
    ss.setdefault("feed_pos", 0); ss.setdefault("feed_on", False); ss.setdefault("feed_sel", None)
    c = st.columns([1, 1, 2.2])
    if c[0].button("⏸️ Pause" if ss.feed_on else "▶️ Start", type="primary", use_container_width=True):
        ss.feed_on = not ss.feed_on
        st.rerun()
    if c[1].button("↺ Reset", use_container_width=True):
        ss.feed_pos, ss.feed_sel, ss.feed_on = 0, None, False
        st.rerun()
    speed = c[2].select_slider("Speed (posts per second)", [2, 5, 10, 25], value=5, label_visibility="collapsed",
                               format_func=lambda v: f"Speed: {v} posts / second")

    def body():
        if ss.feed_on:
            prev = ss.feed_pos
            ss.feed_pos = min(ss.feed_pos + speed, len(src))
            new = src.iloc[prev: ss.feed_pos]
            flagged = new[new["alert"] != SAFE]
            incidents.add_many([{"key": f"post:{r['activity_id']}", "source": "Live feed", "alert": r["alert"], "risk": r["risk"],
                                 "text": r["content"], "sender": acc.loc[r["user_id"], "username"]} for _, r in flagged.iterrows()])
            hot = new[(new["alert"] != SAFE) & (new["risk"] >= 0.75)]
            if not hot.empty:
                h = hot.iloc[-1]
                more = f" (+{len(hot) - 1} more)" if len(hot) > 1 else ""
                st.toast(f"**{h['alert']}** from @{acc.loc[h['user_id'], 'username']}{more}", icon=ICONS[h["alert"]])
        seen = src.iloc[: ss.feed_pos]
        alerts = seen[seen["alert"] != SAFE]
        m = st.columns(3)
        card(m[0], f"{len(seen):,}", "Posts scanned")
        card(m[1], f"{len(alerts):,}", "Threats caught", "#e66767", "Posts flagged as phishing, money scam, spam, misinformation, impersonation or adult scam.")
        card(m[2], f"{alerts['user_id'].nunique():,}", "Attacker accounts", T["primary"], "Distinct accounts that posted at least one flagged message.")
        st.write("")
        left, right = st.columns([1, 1.25])
        with left:
            recent = alerts.iloc[::-1].head(12)
            if recent.empty:
                st.info("Waiting for posts. Press Start.")
            for idx, r in recent.iterrows():
                if alert_row(r["alert"], r["risk"], "@" + acc.loc[r["user_id"], "username"], r["content"], f"f{idx}"):
                    ss.feed_sel = int(idx)
        with right:
            if ss.feed_sel is not None:
                r = src.iloc[ss.feed_sel]
                with st.container(border=True):
                    st.caption(f"Posted {r['timestamp']} · {r['device']} · {r['post_country']}")
                    verdict_view(r["content"], det.analyze(r["content"]), r["user_id"], incident_key=f"post:{r['activity_id']}")
            else:
                st.markdown(f"<div class='card'><div class='l'>Alert details will appear here.</div></div>",
                            unsafe_allow_html=True)

    st.fragment(run_every="1s" if ss.feed_on else None)(body)()


@st.cache_data
def scan_dm_file(mtime):
    d = pd.read_csv(ROOT / "data/dm_messages.csv")
    d["text"] = d["text"].astype(str)
    rng = np.random.default_rng(11)
    genuine = accounts[~accounts["predicted_fake"]]
    senders, receivers = [], []
    for lab in d.get("label", pd.Series([None] * len(d))):
        pool = accounts[accounts["true_type"] == ALERT_TO_ATTACKER[lab]] if lab in ALERT_TO_ATTACKER else genuine
        senders.append(pool["user_id"].iloc[rng.integers(len(pool))])
        receivers.append(genuine["user_id"].iloc[rng.integers(len(genuine))])
    d["sender_id"], d["receiver_id"] = senders, receivers
    return d, det.analyze_many(d["text"])


def page_dm():
    intro("✉️ DM inbox scanner", "Private messages are where most scams happen. The dataset has no DMs, so these were "
          "written by our team and never used for training: a fair test of how the detector handles new messages.")
    ss = st.session_state
    ss.setdefault("live_dms", []); ss.setdefault("dm_sel", 0)

    with st.container(border=True):
        st.markdown("**Try it: send a DM**")
        with st.form("dm", clear_on_submit=True, border=False):
            a, b = st.columns([4, 1])
            txt = a.text_input("Message", placeholder="e.g. Your SBI KYC has expired, update here to avoid account block: sbi-kyc.in",
                               label_visibility="collapsed")
            sent = b.form_submit_button("Send", type="primary", use_container_width=True)
        if sent and txt.strip():
            t = txt.strip()
            v, by, ai = scan(t)
            key = f"live:{h10(t)}:{int(time.time())}"
            log_alert(key, "DM (live demo)", v, t, "you", by)
            ss.live_dms.insert(0, (t, v, by, ai, key))
            ss.dm_sel = 0
    rows = [{"text": t, "v": v, "by": by, "ai": ai, "key": key, "sender": None, "receiver": None, "label": None, "src": "live"}
            for t, v, by, ai, key in ss.live_dms]
    path = ROOT / "data/dm_messages.csv"
    if path.exists():
        d, vs = scan_dm_file(path.stat().st_mtime)
        rows += [{"text": r["text"], "v": v, "by": "Hybrid detector", "ai": None, "key": f"dm:{j}:{h10(r['text'])}",
                  "sender": r["sender_id"], "receiver": r["receiver_id"], "label": r.get("label"), "src": "team"}
                 for j, ((_, r), v) in enumerate(zip(d.iterrows(), vs))]
    if not rows:
        st.info("No messages yet. Send one above, or add data/dm_messages.csv (columns: text, label).")
        return

    blocked = [r for r in rows if r["v"].label != SAFE]
    c = st.columns(4)
    card(c[0], len(rows), "Messages scanned")
    card(c[1], len(blocked), "Blocked as threats", "#e66767")
    card(c[2], sum(r["v"].label == "Money Scam" for r in rows), "Money requests", COLORS["Money Scam"])
    card(c[3], sum(r["v"].label == "Phishing" for r in rows), "Phishing / form requests", COLORS["Phishing"])

    labelled = [r for r in rows if isinstance(r["label"], str)]
    if labelled:
        with st.expander("📊 Test result on the team's messages (ML only vs hybrid)", expanded=False):
            lab = pd.DataFrame({"label": [r["label"] for r in labelled], "hybrid": [r["v"].label for r in labelled],
                                "ml": [r["v"].ml_label for r in labelled]})
            atk = lab["label"] != SAFE
            tbl = pd.DataFrame({
                "Attacks caught": [(lab.loc[atk, k] != SAFE).mean() for k in ("ml", "hybrid")],
                "Correct attack type": [(lab.loc[atk, k] == lab.loc[atk, "label"]).mean() for k in ("ml", "hybrid")],
                "False alarms on safe DMs": [(lab.loc[~atk, k] != SAFE).mean() for k in ("ml", "hybrid")],
            }, index=["ML model only", "Hybrid (rules + ML)"])
            st.dataframe(tbl.style.format("{:.0%}"), use_container_width=True)
            st.caption(f"{int(atk.sum())} attack messages and {int((~atk).sum())} safe messages written by the team.")

    st.write("")
    left, right = st.columns([1, 1.25])
    with left:
        show = st.radio("Show", ["Threats only", "All messages"], horizontal=True, label_visibility="collapsed")
        for i, r in enumerate(rows):
            if show == "Threats only" and r["v"].label == SAFE:
                continue
            who = "You (live demo)" if r["src"] == "live" else f"@{acc.loc[r['sender'], 'username']} → @{acc.loc[r['receiver'], 'username']}"
            if alert_row(r["v"].label, r["v"].risk, who, r["text"], f"dm{i}"):
                ss.dm_sel = i
    with right:
        sel = ss.dm_sel if isinstance(ss.dm_sel, int) else 0
        r = rows[min(sel, len(rows) - 1)]
        v, by, ai = r["v"], r["by"], r["ai"]
        if r["src"] == "team":
            v, by, ai = second_opinion(r["text"], r["v"])
            sender = acc.loc[r["sender"], "username"] if r["sender"] else ""
            log_alert(r["key"], "DM inbox", v, r["text"], sender, by)
        with st.container(border=True):
            verdict_view(r["text"], v, r["sender"], by, ai, r["key"])


def page_analyzer():
    intro("🔍 Message scanner", "Paste any message or post. SocialShield checks it against 20 known scam patterns "
          "and an ML model, then explains its decision.")
    examples = {
        "🎣 Phishing": "Dear user, your Instagram account will be suspended within 24 hours. Verify your login details here: insta-verify-help.net",
        "💸 Money request": "Hi it's Rahul, this is my new number. Lost my phone, can you send 5000 on GPay urgently? Will return tomorrow",
        "📰 Misinformation": "BREAKING: doctors are hiding the truth about this vaccine. Media won't report it. Share before they delete this!",
        "📝 Fake form": "Congratulations! You have won a free iPhone 16. Fill this form with your details to claim your prize: bit.ly/free-ip16",
        "✅ Normal": "Had an amazing time at the beach with family today! The sunset was beautiful.",
    }
    ss = st.session_state
    ss.setdefault("scan_target", "")
    st.markdown("**1. Pick an example** or type your own message below")
    cols = st.columns(len(examples))
    for col, (k, v) in zip(cols, examples.items()):
        if col.button(k, use_container_width=True):
            ss["scan_input"] = v
            ss.scan_target = v
    st.markdown("**2. Press Scan**")
    with st.form("scan_form", border=False):
        st.text_area("Message", key="scan_input", height=100, label_visibility="collapsed",
                     placeholder="Paste or type any message here, e.g. a WhatsApp forward or an Instagram DM")
        if st.form_submit_button("🔍 Scan message", type="primary"):
            ss.scan_target = ss.scan_input
    text = ss.scan_target
    if text.strip():
        st.markdown("**3. Result**")
        v, by, ai = scan(text)
        key = f"scan:{h10(text)}"
        log_alert(key, "Message scanner", v, text, "", by)
        verdict_view(text, v, by=by, ai=ai, incident_key=key)
        with st.expander("Show ML model probabilities"):
            p = pd.Series(v.ml_probs).sort_values()
            fig = go.Figure(go.Bar(x=p.values, y=p.index, orientation="h", marker_color=[COLORS.get(k, "#666") for k in p.index],
                                   hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            fig.update_layout(title="What the ML model alone thinks (before rules)")
            st.plotly_chart(fig_style(fig, 300), use_container_width=True)


def page_accounts():
    intro("👤 Account inspector", "Pick any account to see whether it is genuine or fake, what kind of attacker it is, "
          "and the evidence behind the decision.")
    a, b = st.columns([1, 2.5])
    kind = a.radio("Show", ["Fake accounts", "Genuine accounts"], horizontal=True, label_visibility="collapsed")
    pool = accounts[accounts["predicted_fake"] == (kind == "Fake accounts")].sort_values("fake_prob", ascending=False)
    uid = b.selectbox("Account", pool["user_id"], label_visibility="collapsed",
                      format_func=lambda u: f"@{acc.loc[u, 'username']}  ·  {acc.loc[u, 'fake_prob']:.0%} risk")
    if not uid:
        return
    r = acc.loc[uid]
    fake = bool(r["predicted_fake"])
    label = {v: k for k, v in ALERT_TO_ATTACKER.items()}.get(r["predicted_attack"], SAFE) if fake else SAFE
    c = COLORS[label]
    title = f"FAKE ACCOUNT · {ATTACKER_NAMES[r['predicted_attack']].upper()}" if fake else "GENUINE ACCOUNT"
    st.markdown(f"<div class='banner' style='background:{hexa(c, 0.16)}; border-left:5px solid {c};'>"
                f"<div><div class='t'>{ICONS[label]} {title}</div><div class='s'>@{html.escape(r['username'])} · {html.escape(str(r['full_name']))}</div></div>"
                f"<div style='text-align:right'><div class='r'>{r['fake_prob']:.0%}</div><div class='s'>fake-account risk</div></div></div>",
                unsafe_allow_html=True)
    m = st.columns(4)
    card(m[0], f"{r['account_age_days']}", "days old")
    card(m[1], f"{r['followers_count']:,}", "followers")
    card(m[2], f"{r['following_count']:,}", "following")
    card(m[3], f"{r['posts_count']}", "posts")
    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("**Evidence**")
        if fake and isinstance(r["reasons"], str) and r["reasons"]:
            st.markdown("<ul class='why'>" + "".join(f"<li>{html.escape(x)}</li>" for x in r["reasons"].split(" | ")) + "</ul>",
                        unsafe_allow_html=True)
        else:
            st.caption("Behaviour matches typical genuine accounts.")
    with right:
        if fake and label in ATTACK_INFO:
            st.markdown("**About this attacker**")
            st.markdown(f"{ATTACK_INFO[label]['what']}\n\n**Goal:** {ATTACK_INFO[label]['goal']}")
    up = posts[posts["user_id"] == uid].sort_values("timestamp")
    st.markdown(f"**Posts:** {len(up)} total, {int((up['alert'] != SAFE).sum())} flagged")
    st.dataframe(up[["timestamp", "content", "alert", "risk"]], use_container_width=True, hide_index=True, height=280,
                 column_config={"timestamp": "Posted", "content": st.column_config.TextColumn("Post", width="large"),
                                "alert": "Alert", "risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f")})


@st.cache_data
def cascade(activity_id):
    p = posts.set_index("activity_id").loc[activity_id]
    au = acc.loc[p["user_id"]].copy()
    au["user_id"] = p["user_id"]
    g = simulate_cascade(p, au, accounts[~accounts["true_is_fake"]])
    pos = nx.spring_layout(g, seed=4, k=1 / max(np.sqrt(g.number_of_nodes()), 1))
    nodes = sorted(g.nodes(data=True), key=lambda nd: nd[1]["role"] == "source")
    ex, ey = [], []
    for u, v in g.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    pts = [(pos[n][0], pos[n][1], d) for n, d in nodes]
    return {"stats": cascade_stats(g), "reach": reach_over_time(g), "edges": (ex, ey), "nodes": pts}


def network_fig(c, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["edges"][0], y=c["edges"][1], mode="lines", line=dict(color=T["edge"], width=1), hoverinfo="skip"))
    xs, ys, ds = zip(*c["nodes"])
    src = [d["role"] == "source" for d in ds]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=[30 if s else 5 + 1.6 * np.log1p(d["followers"]) for s, d in zip(src, ds)],
                    color=[color if s else T["node"] for s in src],
                    line=dict(width=[3 if s else 1 for s in src], color=[T["text"] if s else T["bg"] for s in src])),
        text=[f"@{d['username']}<br>{d['followers']:,} followers<br>{d['t']:.1f} h after posting" for d in ds],
        hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig_style(fig, 400)


def page_spread():
    intro("🕸️ How misinformation spreads", "Each dot is a person who reshared the post. The large dot is the original poster. "
          "Reshare counts and resharer accounts are real data; who shared from whom is simulated the same way for both posts.")
    mis = posts[posts["alert"] == "Misinformation"].sort_values("shares", ascending=False).drop_duplicates("content").head(25)
    pick = st.selectbox("Misinformation post", mis["activity_id"], label_visibility="collapsed",
                        format_func=lambda i: f"{int(mis.set_index('activity_id').loc[i, 'shares'])} shares · {mis.set_index('activity_id').loc[i, 'content'][:95]}")
    mp = mis.set_index("activity_id").loc[pick]
    f0 = acc.loc[mp["user_id"], "followers_count"]
    real = accounts.loc[~accounts["true_is_fake"], ["user_id", "followers_count"]]
    near = real.iloc[(real["followers_count"] - f0).abs().argsort()[:40]]["user_id"]
    normal = posts[posts["user_id"].isin(near) & (posts["alert"] == SAFE)].sort_values("shares", ascending=False).iloc[0]
    c1, c2 = cascade(pick), cascade(normal["activity_id"])
    cols = st.columns(2)
    for col, c, text, title, color in [(cols[0], c1, mp["content"], "📰 Misinformation post", COLORS["Misinformation"]),
                                       (cols[1], c2, normal["content"], "✅ Normal post, similar-sized author", COLORS[SAFE])]:
        with col:
            st.markdown(f"**{title}**<br><span class='small'>{html.escape(text)}</span>", unsafe_allow_html=True)
            m = st.columns(3)
            card(m[0], c["stats"]["reshares"], "reshares", color)
            card(m[1], f"{c['stats']['potential_reach']:,}", "people reached*", color, "Simulated: total followers of everyone who reshared. Shows scale, not a measured number.")
            card(m[2], c["stats"]["depth"], "levels deep", color, "How many reshare steps away from the original poster it travelled.")
            st.plotly_chart(network_fig(c, color), use_container_width=True)
    fig = go.Figure()
    for c, name, color in [(c1, "Misinformation post", COLORS["Misinformation"]), (c2, "Normal post", COLORS[SAFE])]:
        fig.add_trace(go.Scatter(x=c["reach"]["hour"], y=c["reach"]["reach"], name=name, line=dict(color=color, width=2.5)))
    fig.update_layout(title="People reached in the first 48 hours*", hovermode="x unified", legend=dict(orientation="h", y=-0.22),
                      xaxis_title="Hours after posting", yaxis_title="People reached")
    st.plotly_chart(fig_style(fig, 330), use_container_width=True)
    st.caption("*Simulated: sum of resharers' followers. Use reshare counts, not reach, as the measured result.")


@st.cache_data
def evasion_curve(changed):
    fake_ids = accounts.loc[accounts["in_test_set"] & accounts["true_is_fake"], "user_id"]
    real_ids = accounts.loc[~accounts["in_test_set"] & ~accounts["true_is_fake"], "user_id"]
    med, fx, model = feats.loc[real_ids].median(), feats.loc[fake_ids], account_model()
    out = {}
    for a in np.round(np.linspace(0, 1, 21), 2):
        x = fx.copy()
        if changed:
            x[list(changed)] = fx[list(changed)] + a * (med[list(changed)] - fx[list(changed)])
        out[float(a)] = model.predict(x[feats.columns]) == 1
    types = accounts.set_index("user_id").loc[fake_ids, "true_type"].map(ATTACKER_NAMES).values
    return out, types


def page_evasion():
    intro("🧪 Evasion lab", "Attackers adapt. Drag the slider to make fake accounts behave more like a typical genuine "
          "user, and watch how many the model still catches.")
    with st.expander("Choose what the attacker changes"):
        changed = st.multiselect("Features", list(MIMICABLE), default=list(MIMICABLE), format_func=MIMICABLE.get,
                                 label_visibility="collapsed")
    alpha = st.slider("Imitation level", 0, 100, 0, 5, format="%d%%") / 100
    curve, types = evasion_curve(tuple(sorted(changed)))
    caught = curve[round(alpha, 2)]
    m = st.columns(3)
    card(m[0], f"{caught.mean():.0%}", "fake accounts still caught", T["primary"])
    card(m[1], f"{(~caught).sum()} / {len(caught)}", "slip through undetected", "#e66767")
    card(m[2], f"{alpha:.0%}", "imitation level", "0% = fake accounts behave as they do in the data. 100% = they copy a typical genuine account completely.")
    xs = sorted(curve)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x * 100 for x in xs], y=[curve[x].mean() * 100 for x in xs], mode="lines+markers",
                             line=dict(color=T["primary"], width=2.5), marker=dict(size=7), name="Detection rate",
                             hovertemplate="Imitation %{x:.0f}%: %{y:.0f}% caught<extra></extra>"))
    fig.add_trace(go.Scatter(x=[alpha * 100], y=[caught.mean() * 100], mode="markers", name="You are here",
                             marker=dict(size=18, color=COLORS["Money Scam"], line=dict(width=2, color=T["bg"]))))
    fig.update_layout(title="Detection rate as fake accounts imitate real users", yaxis_range=[-5, 105],
                      xaxis_title="Imitation of a typical genuine account (%)", yaxis_title="Fake accounts caught (%)",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_style(fig, 380), use_container_width=True)
    by = pd.Series(caught).groupby(types).mean().sort_values()
    with st.expander("Detection by attacker type at this level"):
        st.dataframe(by.map("{:.0%}".format).rename("Caught").to_frame(), use_container_width=True)
    st.info("**Takeaway:** perfect against careless bots, but detection collapses once bots copy genuine behaviour. "
            "Real platforms respond by retraining on new attacks and adding signals that are hard to fake, "
            "such as device fingerprints and network-level patterns.")


def page_incidents():
    intro("📒 Incident log", "Every alert SocialShield raised, and what was done about it: <b>detect → respond → record</b>, "
          "like a security operations centre. Use the buttons to block, quarantine or dismiss.")
    inc = incidents.all_incidents()
    if not inc:
        st.info("No incidents yet. Start the Live Feed, open the DM Inbox or scan a message.")
        return
    df = pd.DataFrame(inc)
    k = st.columns(4)
    card(k[0], len(df), "Total incidents")
    card(k[1], int((df["status"] == incidents.OPEN).sum()), "Open", "#e66767")
    card(k[2], int(df["status"].isin([incidents.ACTIONS["block"], incidents.ACTIONS["quarantine"]]).sum()), "Contained", COLORS["Phishing"])
    card(k[3], int((df["status"] == incidents.ACTIONS["dismiss"]).sum()), "False alarms", COLORS[SAFE])
    st.write("")
    f1, f2, f3 = st.columns(3)
    stat = f1.multiselect("Status", sorted(df["status"].unique()), placeholder="All statuses")
    typ = f2.multiselect("Attack type", sorted(df["alert"].unique()), placeholder="All attack types")
    srcs = f3.multiselect("Source", sorted(df["source"].unique()), placeholder="All sources")
    mask = pd.Series(True, index=df.index)
    for col, chosen in [("status", stat), ("alert", typ), ("source", srcs)]:
        if chosen:
            mask &= df[col].isin(chosen)
    view = df[mask].sort_values("id", ascending=False)
    tab1, tab2 = st.tabs(["Act on incidents", "Table and export"])
    with tab1:
        st.caption(f"Showing {min(len(view), 25)} of {len(view)} (newest first). Block = stop the sender · Quarantine = hide the message · Not a threat = false alarm")
        for i in view.head(25).to_dict("records"):
            incident_row(i, "log")
    with tab2:
        cols = ["id", "time", "source", "alert", "risk", "status", "action_time", "sender", "detected_by", "text"]
        st.dataframe(view[cols], use_container_width=True, hide_index=True, height=420,
                     column_config={"risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f"),
                                    "text": st.column_config.TextColumn("Message", width="large")})
        a, b = st.columns([1, 1])
        a.download_button("⬇️ Download as CSV", view[cols].to_csv(index=False).encode("utf-8"),
                          "socialshield_incidents.csv", "text/csv", use_container_width=True)
        with b.popover("🗑️ Clear the log", use_container_width=True):
            st.write("This deletes every incident.")
            if st.button("Yes, clear everything", type="primary"):
                incidents.clear()
                st.cache_resource.clear()
                st.rerun()


# ================================================================ NAVIGATION
P = {"home": st.Page(page_home, title="Start here", icon="🏠", default=True),
     "overview": st.Page(page_overview, title="Overview", icon="📊", url_path="overview"),
     "feed": st.Page(page_feed, title="Live Threat Feed", icon="🚨", url_path="feed"),
     "dms": st.Page(page_dm, title="DM Inbox", icon="✉️", url_path="dms"),
     "scan": st.Page(page_analyzer, title="Message Scanner", icon="🔍", url_path="scan"),
     "accounts": st.Page(page_accounts, title="Account Inspector", icon="👤", url_path="accounts"),
     "spread": st.Page(page_spread, title="Spread Network", icon="🕸️", url_path="spread"),
     "evasion": st.Page(page_evasion, title="Evasion Lab", icon="🧪", url_path="evasion"),
     "incidents": st.Page(page_incidents, title="Incident Log", icon="📒", url_path="incidents")}
nav = st.navigation({
    "": [P["home"]],
    "Monitor": [P["overview"], P["feed"], P["dms"]],
    "Investigate": [P["scan"], P["accounts"]],
    "Respond": [P["incidents"]],
    "Research": [P["spread"], P["evasion"]],
})
@st.cache_resource
def seed_dm_alerts(mtime):
    """Incoming DMs flagged by the detector become open alerts as soon as the app starts."""
    d, vs = scan_dm_file(mtime)
    incidents.add_many([{"key": f"dm:{j}:{h10(r.text)}", "source": "DM inbox", "alert": v.label, "risk": v.risk,
                         "text": r.text, "sender": acc.loc[r.sender_id, "username"]}
                        for j, (r, v) in enumerate(zip(d.itertuples(), vs)) if v.label != SAFE])
    return True


if (ROOT / "data/dm_messages.csv").exists():
    seed_dm_alerts((ROOT / "data/dm_messages.csv").stat().st_mtime)

with st.sidebar:
    n_open = sum(i["status"] == incidents.OPEN for i in incidents.all_incidents())
    st.page_link(P["incidents"], label=f"**{n_open} open alerts**", icon="🔔")
    ready, model = ai_status()
    st.session_state.setdefault("ai_on", True)
    st.toggle("🤖 AI second opinion", key="ai_on", disabled=not ready,
              help="A local AI model double-checks messages the rules and ML model think are safe. "
                   "Runs on this laptop with Ollama; nothing is sent to the internet.")
    st.caption(f"Local model: {model}" if ready else f"AI layer offline: {model}")
    st.divider()
    st.caption("Network Security Project · MPSTME\n\nData: Kaggle (synthetic). DMs and spread paths simulated.")
nav.run()
