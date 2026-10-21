"""Hybrid message detector: signature rules + ML model.

Works like a hybrid intrusion detection system:
  * Signature layer: hand-written rules for known scam patterns (like Snort rules in an IDS).
  * ML layer: the trained text model, which catches patterns learned from the dataset.
The final alert combines both, and every alert carries its evidence so the app can explain it.

IMPORTANT: these rules were written from general knowledge of social engineering attacks,
BEFORE the team's DM test set was created, so the DM test is a fair test.
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import clean_text  # noqa: E402

SAFE = "Safe"

# (rule id, alert category, weight, regex, human explanation)
RULES = [
    # Phishing / credential theft
    ("PH-01", "Phishing", 0.35, r"\b(verify|re-?authenticate|confirm|update|validate)\b.{0,40}\b(account|details|identity|password|login|kyc|information)\b",
     "Asks you to verify or update account details"),
    ("PH-02", "Phishing", 0.35, r"\b(password|otp|one[- ]time password|pin|cvv|login credentials|security code)\b",
     "Mentions passwords, OTP or PIN"),
    ("PH-03", "Phishing", 0.30, r"\b(suspend(ed)?|locked|deactivat\w*|blocked|terminated|disabled)\b",
     "Threatens that your account will be suspended or locked"),
    ("PH-04", "Phishing", 0.30, r"\b(fill|submit|complete)\b.{0,25}\b(form|survey|application|details)\b",
     "Asks you to fill in a form with your details"),
    ("PH-05", "Phishing", 0.25, r"\b(won|winner|congratulations|claim|prize|reward|gift card|lottery)\b",
     "Prize or reward bait"),
    ("PH-06", "Phishing", 0.15, r"\bkyc\b", "Mentions KYC (common in Indian banking scams)"),
    # Money requests and financial scams
    ("MS-01", "Money Scam", 0.40, r"\b(send|transfer|pay|lend|deposit)\b.{0,30}(\b(money|rs\.?|rupees|inr|usd|dollars|amount|funds)\b|₹|\$|\b\d{3,}\b)",
     "Asks you to send or transfer money"),
    ("MS-02", "Money Scam", 0.35, r"\b(gpay|google pay|phonepe|paytm|upi|bhim|paypal|venmo|western union|gift ?card|bitcoin|btc|usdt|crypto wallet|wallet address)\b",
     "Mentions a payment app or crypto wallet"),
    ("MS-03", "Money Scam", 0.30, r"\b(double|triple|\d+x)\b.{0,30}\b(money|investment|returns?|profit)\b|\bguaranteed (returns?|profit)\b|\b(invest|investment)\b.{0,40}\b(profit|returns?)\b",
     "Promises unrealistic investment returns"),
    ("MS-04", "Money Scam", 0.20, r"\b(emergency|hospital|stuck|accident|stranded)\b.{0,60}\b(money|send|pay|help)\b|\b(money|send|pay)\b.{0,60}\b(emergency|hospital|stuck|accident|stranded)\b",
     "Emergency story used to pressure you into paying"),
    # Misinformation
    ("MI-01", "Misinformation", 0.35, r"\bshare (this )?before (it'?s |they )?(deleted|removed|delete|take)",
     "Urges sharing before it gets deleted"),
    ("MI-02", "Misinformation", 0.30, r"\b(they|media|government|doctors|mainstream)\b.{0,30}\b(don'?t want you to know|hiding|won'?t (tell|report)|cover(ing)? up)\b",
     "Claims a cover-up by media or authorities"),
    ("MI-03", "Misinformation", 0.20, r"\b(research it yourself|do your own research|wake up|100% (true|proof|proven)|exposed)\b",
     "Uses conspiracy-style persuasion phrases"),
    ("MI-04", "Misinformation", 0.15, r"^\s*(breaking|fact|urgent news|leaked)\b",
     "Starts with sensational framing (BREAKING, FACT, LEAKED)"),
    # Spam
    ("SP-01", "Spam", 0.25, r"\b(dm me|message me|inbox me|click (the )?link|link in bio|check (my|the) bio)\b",
     "Pushes you to a DM or external link"),
    ("SP-02", "Spam", 0.20, r"\b(lose \d+ (pounds|kg)|work from home|earn \S+ (daily|per day|a day|weekly)|limited (time )?offer|only \d+ (spots|people))\b",
     "Classic spam offer wording"),
    # Impersonation
    ("IM-01", "Impersonation", 0.30, r"\b(official|verified)\b.{0,20}\b(account|page|support|team|admin)\b|\b(this is|i am|i'm) (the )?(official|admin|support)\b",
     "Claims to be an official account or support team"),
    ("IM-02", "Impersonation", 0.25, r"\b(new (number|account)|my old account (was )?hacked|lost my phone)\b",
     "Says it is a known person on a new account (common impersonation trick)"),
    # Shared amplifiers (no category of their own)
    ("GN-01", None, 0.15, r"\b(urgent(ly)?|immediately|right now|asap|within \d+ (hours|minutes)|act fast|last chance|today only)\b",
     "Creates urgency or time pressure"),
    ("GN-02", None, 0.15, r"(https?://|www\.|bit\.ly|tinyurl|\b[\w-]+\.(com|net|info|org|io|ru|xyz|in|link|top)\b)",
     "Contains a link"),
]
COMPILED = [(rid, cat, w, re.compile(rx, re.I), why) for rid, cat, w, rx, why in RULES]

ATTACK_INFO = {
    "Phishing": {
        "what": "The attacker pretends to be a platform or bank and sends a link to a fake login page or form.",
        "goal": "Steal passwords, OTPs or personal details, then take over the victim's accounts.",
        "action": "Do not click the link or enter details. Report the sender and visit the real site directly.",
    },
    "Money Scam": {
        "what": "The attacker asks for money directly, or lures the victim with fake investment or crypto profits.",
        "goal": "Get the victim to transfer money that cannot be recovered.",
        "action": "Never send money from a message alone. Verify the person through another channel first.",
    },
    "Spam": {
        "what": "Automated accounts post or send the same promotional message to many users.",
        "goal": "Drive traffic to scam sites, malware downloads or paid promotions.",
        "action": "Ignore and report. Do not open the links.",
    },
    "Misinformation": {
        "what": "Sensational false claims written to trigger anger or fear so people share them quickly.",
        "goal": "Spread false narratives and grow reach for the accounts behind them.",
        "action": "Do not share. Check the claim with reliable news or fact-checking sources.",
    },
    "Impersonation": {
        "what": "The attacker poses as a celebrity, brand, official account or someone you know.",
        "goal": "Use borrowed trust to run scams or collect information.",
        "action": "Check the verified badge, account age and follower history before trusting it.",
    },
    "Adult Scam": {
        "what": "Sexual bait that pushes users to an external link or private page.",
        "goal": "Lead victims to paid scams, malware or blackmail (sextortion).",
        "action": "Do not click. Block and report the account.",
    },
}


@dataclass
class Verdict:
    label: str
    risk: float                      # 0 to 1
    ml_label: str
    ml_probs: dict
    rule_hits: list = field(default_factory=list)   # dicts: id, category, why, matches
    ml_words: list = field(default_factory=list)    # words the ML model weighted most
    info: dict = field(default_factory=dict)


class MessageDetector:
    def __init__(self, model_path=ROOT / "models/post_model.joblib"):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes = list(self.model.classes_)
        union = self.model.steps[0][1]
        self.word_vec = dict(union.transformer_list)["words"]
        self.n_word = len(self.word_vec.vocabulary_)
        self.word_names = self.word_vec.get_feature_names_out()
        self.clf = self.model.steps[-1][1]

    def _rules(self, text):
        hits = []
        for rid, cat, w, rx, why in COMPILED:
            found = [m.group(0) for m in rx.finditer(text)]
            if found:
                hits.append({"id": rid, "category": cat, "weight": w, "why": why, "matches": found})
        return hits

    def _top_words(self, text, label, k=5):
        if label == SAFE:
            return []
        x = self.word_vec.transform([text]).toarray()[0]
        coef = self.clf.coef_[self.classes.index(label)][: self.n_word]
        contrib = x * coef
        idx = np.argsort(contrib)[::-1][: k * 4]
        return [self.word_names[i] for i in idx if contrib[i] > 0 and not self._filler(self.word_names[i])][:k]

    @staticmethod
    def _filler(term):
        # Hide placeholder tokens and words like "is", "my" that carry no meaning for a reader.
        return "token" in term or all(w in ENGLISH_STOP_WORDS or len(w) < 3 for w in term.split())

    def analyze(self, text: str) -> Verdict:
        return self._combine(text, dict(zip(self.classes, self.model.predict_proba([text])[0])))

    def analyze_many(self, texts) -> list:
        """Score many messages at once (one batched ML call, then rules per message)."""
        texts = list(texts)
        probs = self.model.predict_proba(texts)
        return [self._combine(t, dict(zip(self.classes, p))) for t, p in zip(texts, probs)]

    def _combine(self, text: str, probs: dict) -> Verdict:
        ml_label = max(probs, key=probs.get)
        hits = self._rules(text)

        # Score each attack category: ML probability plus the weight of matching rules.
        amplifier = sum(h["weight"] for h in hits if h["category"] is None)
        scores = {}
        for cat in ATTACK_INFO:
            rule_score = sum(h["weight"] for h in hits if h["category"] == cat)
            scores[cat] = probs.get(cat, 0) + rule_score + (amplifier if rule_score > 0 else 0)
        best = max(scores, key=scores.get)
        risk = min(scores[best], 1.0)

        # Raise an alert if the combined evidence is strong enough.
        label = best if risk >= 0.5 else SAFE
        if label == SAFE:
            risk = min(1 - probs.get(SAFE, 0) + amplifier * 0.5, 0.49)
        return Verdict(
            label=label, risk=round(float(risk), 3), ml_label=ml_label,
            ml_probs={k: round(float(v), 3) for k, v in probs.items()},
            rule_hits=hits, ml_words=self._top_words(text, best),
            info=ATTACK_INFO.get(label, {}),
        )


if __name__ == "__main__":
    d = MessageDetector()
    for msg in [
        "Hey bro send me 2000 rs on gpay urgently, I will return tomorrow",
        "BREAKING: government hiding the truth about vaccines, share before they delete this!",
        "Your Instagram account will be suspended. Verify your details here: insta-help-center.net",
        "Hi, this is the official Amazon support team. Please fill this form to get your refund: amzn-refunds.xyz",
        "Had a great time at the beach with family today!",
        "Can you send me the notes from today's lecture?",
        "Congrats on the new job! Let's celebrate this weekend",
        "Hi it's Rahul, this is my new number. Lost my phone, can you pay 5000 for me urgently? Will return",
    ]:
        v = d.analyze(msg)
        print(f"{v.label:15s} risk={v.risk:.2f} ml={v.ml_label:14s} rules={[h['id'] for h in v.rule_hits]}  | {msg[:60]}")
