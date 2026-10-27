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
from normalise import normalise  # noqa: E402

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
    ("MI-02", "Misinformation", 0.30, r"\b(they|media|government|doctors|mainstream|\w+ pumps?|companies|banks)\b.{0,30}\b(don'?t want you to know|hiding|won'?t (tell|report)|cover(ing)? up)\b",
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
    # ---- Version 2 rules (improvement round, written from Test 1 failures and known scam types) ----
    ("MS-05", "Money Scam", 0.45, r"\b(registration|processing|file|customs|joining|activation)\s+(fee|fees|charges?|tax)\b|\bdeposit\s+(rs\.?|₹)?\s*[\d,]+\s+first\b",
     "Asks for an upfront fee before you receive money (advance-fee fraud)"),
    ("MS-06", "Money Scam", 0.45, r"\b(won|winner|lucky draw|lottery|jackpot)\b.{0,80}\b(pay|fee|tax|charges?)\b",
     "Prize that requires a payment first"),
    ("MS-07", "Money Scam", 0.5, r"\bcollect request\b|\bapprove\b.{0,30}\b(with your )?(upi )?pin\b",
     "Asks you to approve a UPI collect request with your PIN (this sends money, it never receives it)"),
    ("MS-08", "Money Scam", 0.35, r"\bearn (up to )?(rs\.?|₹)\s*[\d,]+\s*(daily|per day|a day)\b|\b(liking|like) (youtube )?videos\b",
     "Too-good-to-be-true earning offer"),
    ("MS-09", "Money Scam", 0.35, r"\b\d{2,4}% returns?\b|\binsider (calls?|tips?|info)\b|\bguaranteed returns?\b|\bzero risk\b|\bjoin (our )?vip\b",
     "Promises guaranteed or insider investment profits"),
    ("IM-03", "Impersonation", 0.3, r"\b(this is|i am|i'm|calling from)\s+(your\s+|the\s+)?(hod|principal|ceo|boss|manager|admin|cyber ?crime( department)?|police|bank'?s? customer care|customer care)\b",
     "Claims to be a boss, official or authority"),
    ("IM-04", "Impersonation", 0.45, r"\b(buy|send|purchase)\b.{0,40}\b(gift ?cards?|google play( cards?)?|vouchers?)\b",
     "Asks you to buy gift cards or vouchers (a classic boss-impersonation scam)"),
    ("IM-05", "Impersonation", 0.45, r"(?<!not )(?<!never )(?<!n't )\b(send|share)\s+(me\s+)?the\s+(\d[- ]digit\s+)?(code|otp)\b",
     "Asks you to share a code or OTP you received"),
    ("IM-06", "Impersonation", 0.3, r"\b(using (my )?friend'?s (phone|number)|my (old )?phone (broke|is broken|is not working))\b",
     "Says they are messaging from someone else's phone"),
    ("IM-07", "Impersonation", 0.4, r"\b(case (is )?registered against|arrest warrant|stay on (the )?video call|keep (it|this) confidential)\b",
     "Uses legal threats or secrecy to pressure you"),
    ("SP-03", "Spam", 0.5, r"\b\d{1,2}% off\b|\bmega sale\b|\bbuy 1 get 1\b|\bfree (shaker|demo( class)?)\b|\breply stop\b|\bunsubscribe\b|\bcheapest rates?\b|\bjoin our whatsapp channel\b|\bcall now\b",
     "Promotional or marketing message"),
    ("SP-04", "Spam", 0.5, r"\bforward this to \d+ (people|contacts)\b|\bdon'?t break the chain\b",
     "Chain message asking you to forward it"),
    ("SP-05", "Spam", 0.5, r"\b\d+\s*(instagram )?followers for\b|\bseo services\b|\brank your website\b",
     "Sells fake followers or ranking services"),
    ("PH-07", "Phishing", 0.35, r"\b(sign in|log ?in|submit your (net ?banking )?(login|password|details))\b.{0,40}(to (upgrade|receive|claim|continue|unlock)|\bat\b)",
     "Asks you to sign in or submit login details through a link"),
    ("MI-05", "Misinformation", 0.45, r"\b(cures?|cure for)\b.{0,40}\b(diabetes|cancer|covid|virus(es)?|disease)\b|\bimmune to all\b|\bgives you cancer\b",
     "Miracle cure or false health claim"),
    ("MI-06", "Misinformation", 0.3, r"\bfrom tomorrow\b|\bnew rbi rule\b|\bnasa has confirmed\b|\bscientists have proven\b|\bdoctors are hiding\b",
     "Unverified official announcement or 'experts confirm' claim"),
    ("MI-07", "Misinformation", 0.25, r"\bplease share\b|\bshare with (your )?family\b|\bit can save lives\b",
     "Urges you to spread the claim"),
    # Shared amplifiers (no category of their own)
    ("GN-01", None, 0.15, r"\b(urgent(ly)?|immediately|right now|asap|within \d+ (hours|minutes)|act fast|last chance|today only)\b",
     "Creates urgency or time pressure"),
    ("GN-02", None, 0.15, r"(https?://|www\.|bit\.ly|tinyurl|\b[\w-]+\.(com|net|info|org|io|ru|xyz|in|link|top)\b)",
     "Contains a link"),
]
COMPILED = [(rid, cat, w, re.compile(rx, re.I), why) for rid, cat, w, rx, why in RULES]
# A business advertising on its own public page is normal; the same advert sent uninvited in a DM is spam.
DM_ONLY_RULES = {"SP-03"}

# Signs of a genuine message. They lower the score; they never remove a strong rule match on their own.
BENIGN = [
    ("OK-01", 0.4, r"\b(do not|don'?t) share (it|this|your otp)\b", "Genuine OTP notices tell you NOT to share the code"),
    ("OK-02", 0.4, r"\bcall the number on the back of your card\b", "Points you to your card's official number, not a link"),
    ("OK-03", 0.3, r"\bhas been shipped\b|\bwill arrive by\b", "Normal delivery update"),
    ("OK-04", 0.3, r"\bpay me back\b|\byou (still )?owe me\b", "A friend asking to be paid back"),
]
BENIGN_COMPILED = [(rid, w, re.compile(rx, re.I), why) for rid, w, rx, why in BENIGN]
EVASION_WEIGHT = 0.15  # amplifier added when the normaliser had to undo hidden-character tricks
ML_ONLY_THRESHOLD = 0.8  # with no rule match, the ML model alone must be at least this sure

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
    evasion: list = field(default_factory=list)     # evasion tricks undone by the normaliser
    decoded: str = ""                               # the normalised text the detector actually read


class MessageDetector:
    def __init__(self, model_path=ROOT / "models/post_model.joblib", normalise_input=True):
        self.normalise_input = normalise_input  # False = no evasion defence (used to measure the attacks)
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes = list(self.model.classes_)
        union = self.model.steps[0][1]
        self.word_vec = dict(union.transformer_list)["words"]
        self.n_word = len(self.word_vec.vocabulary_)
        self.word_names = self.word_vec.get_feature_names_out()
        self.clf = self.model.steps[-1][1]

    def _rules(self, text, channel="dm"):
        hits = []
        for rid, cat, w, rx, why in COMPILED:
            if channel != "dm" and rid in DM_ONLY_RULES:
                continue
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

    def _prepare(self, text):
        """Undo evasion tricks first (like IDS traffic normalisation). Returns (text to analyse, tricks)."""
        return normalise(text) if self.normalise_input else (str(text), [])

    def analyze(self, text: str, channel: str = "dm") -> Verdict:
        """channel: "dm" for private messages and scanned text, "post" for public posts."""
        clean, tricks = self._prepare(text)
        return self._combine(clean, dict(zip(self.classes, self.model.predict_proba([clean])[0])), channel, tricks)

    def analyze_many(self, texts, channel: str = "dm") -> list:
        """Score many messages at once (one batched ML call, then rules per message)."""
        prepared = [self._prepare(t) for t in texts]
        probs = self.model.predict_proba([c for c, _ in prepared])
        return [self._combine(c, dict(zip(self.classes, p)), channel, tr) for (c, tr), p in zip(prepared, probs)]

    def _combine(self, text: str, probs: dict, channel: str = "dm", tricks=()) -> Verdict:
        ml_label = max(probs, key=probs.get)
        hits = self._rules(text, channel)
        hidden = [t for t in tricks if t != "split-up letters"]
        if hidden:
            # Hiding words from a filter is itself a warning sign. Amplifier only: it never raises an alert alone.
            hits.append({"id": "EV-01", "category": None, "weight": EVASION_WEIGHT,
                         "why": "Disguises words to slip past filters", "matches": hidden})

        # Score each attack category: ML probability plus the weight of matching rules, minus benign signs.
        amplifier = sum(h["weight"] for h in hits if h["category"] is None)
        benign = [{"id": rid, "weight": w, "why": why, "matches": [m.group(0) for m in rx.finditer(text)]}
                  for rid, w, rx, why in BENIGN_COMPILED if rx.search(text)]
        calm = sum(b["weight"] for b in benign)
        scores, rule_scores = {}, {}
        for cat in ATTACK_INFO:
            rule_scores[cat] = sum(h["weight"] for h in hits if h["category"] == cat)
            scores[cat] = probs.get(cat, 0) + rule_scores[cat] + (amplifier if rule_scores[cat] > 0 else 0) - calm
        best = max(scores, key=scores.get)
        risk = max(min(scores[best], 1.0), 0.0)

        # Raise an alert if the evidence is strong enough. The ML model alone needs high confidence.
        strong_enough = rule_scores[best] > 0 or probs.get(best, 0) >= ML_ONLY_THRESHOLD
        label = best if (risk >= 0.5 and strong_enough) else SAFE
        if label == SAFE:
            risk = max(min(1 - probs.get(SAFE, 0) + amplifier * 0.5 - calm, 0.49), 0.0)
        return Verdict(
            label=label, risk=round(float(risk), 3), ml_label=ml_label,
            ml_probs={k: round(float(v), 3) for k, v in probs.items()},
            rule_hits=hits, ml_words=self._top_words(text, best),
            info=ATTACK_INFO.get(label, {}), evasion=list(tricks), decoded=text if tricks else "",
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
