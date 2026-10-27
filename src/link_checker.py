"""Link checker: inspect every link or domain in a message, the way a secure email gateway does.

Checks (each explained in plain words in the app):
  LK-01  Domain is on a real public phishing blocklist (Phishing.Database snapshot)
  LK-02  A known brand's name or a look-alike of it (paypa1, g00gle, amaz0n) on a domain that brand does not own
  LK-03  Internationalised domain (punycode "xn--" or non-Latin letters): can imitate a real domain letter for letter
  LK-04  Raw IP address instead of a domain name
  LK-05  Domain ending often abused by scammers (.top, .xyz, .tk, ...)
  LK-06  Free hosting or site-builder subdomain often used for throwaway phishing pages
  LK-07  Login or payment words inside the domain name (secure-, verify-, kyc-, billing-)
  LK-08  Link shortener that hides the real destination
Heuristics LK-02 to LK-08 need no internet and no list, so they also work on brand-new phishing domains
that no blocklist has seen yet. The blocklist (LK-01) is optional: if data/external/phishing_domains.txt.gz
is missing, the checker still runs.
"""
import gzip
import ipaddress
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKLIST = ROOT / "data/external/phishing_domains.txt.gz"

# Brand name -> domains the brand really owns (registered domains)
BRANDS = {
    "google": {"google.com", "google.co.in", "youtube.com", "gmail.com", "goo.gl", "g.co", "googleusercontent.com", "googleapis.com",
               "googleadservices.com", "googlesyndication.com", "googlevideo.com", "gstatic.com", "googletagmanager.com"},
    "gmail": {"gmail.com", "google.com"},
    "youtube": {"youtube.com", "youtu.be"},
    "facebook": {"facebook.com", "fb.com", "fb.me", "meta.com"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com", "wa.me"},
    "microsoft": {"microsoft.com", "live.com", "outlook.com", "office.com", "microsoftonline.com", "microsoftstore.com", "msn.com", "bing.com", "azure.com"},
    "outlook": {"outlook.com", "live.com", "microsoft.com"},
    "office365": {"office.com", "microsoft.com"},
    "apple": {"apple.com", "icloud.com"},
    "icloud": {"icloud.com", "apple.com"},
    "amazon": {"amazon.com", "amazon.in", "amzn.to", "amazon.co.uk", "amazonaws.com", "images-amazon.com", "media-amazon.com",
               "ssl-images-amazon.com", "amazontrust.com"},
    "netflix": {"netflix.com"},
    "paypal": {"paypal.com", "paypal.me"},
    "flipkart": {"flipkart.com"},
    "myntra": {"myntra.com"},
    "paytm": {"paytm.com", "paytm.in"},
    "phonepe": {"phonepe.com"},
    "gpay": {"google.com", "pay.google.com"},
    "sbi": {"sbi.co.in", "onlinesbi.sbi", "onlinesbi.com", "sbicard.com"},
    "onlinesbi": {"onlinesbi.sbi", "onlinesbi.com", "sbi.co.in"},
    "hdfc": {"hdfcbank.com", "hdfc.com"},
    "hdfcbank": {"hdfcbank.com"},
    "icici": {"icicibank.com", "icici.com"},
    "icicibank": {"icicibank.com"},
    "axisbank": {"axisbank.com"},
    "kotak": {"kotak.com"},
    "uidai": {"uidai.gov.in"},
    "aadhaar": {"uidai.gov.in"},
    "incometax": {"incometax.gov.in"},
    "irctc": {"irctc.co.in"},
    "epfo": {"epfindia.gov.in"},
    "airtel": {"airtel.in", "airtel.com"},
    "jio": {"jio.com"},
    "dhl": {"dhl.com", "dhl.co.in"},
    "fedex": {"fedex.com"},
    "indiapost": {"indiapost.gov.in"},
    "linkedin": {"linkedin.com", "lnkd.in"},
    "twitter": {"twitter.com", "x.com", "t.co"},
    "telegram": {"telegram.org", "t.me"},
    "steam": {"steampowered.com", "steamcommunity.com"},
    "binance": {"binance.com"},
    "coinbase": {"coinbase.com"},
    "metamask": {"metamask.io"},
    "dropbox": {"dropbox.com"},
    "docusign": {"docusign.com", "docusign.net"},
    "adobe": {"adobe.com"},
    "wellsfargo": {"wellsfargo.com"},
    "chase": {"chase.com"},
    "bankofamerica": {"bankofamerica.com"},
    "usps": {"usps.com"},
    "yahoo": {"yahoo.com"},
    "office": {"office.com", "microsoft.com"},
    "spotify": {"spotify.com"},
    "hotstar": {"hotstar.com"},
    "swiggy": {"swiggy.com"},
    "zomato": {"zomato.com"},
}
# Short brand names match only as a whole word inside the domain, to avoid "chase" in "purchase" etc.
SHORT_BRANDS = {"sbi", "jio", "dhl", "hdfc", "icici", "kotak", "usps", "chase", "steam", "apple", "office", "adobe", "gpay", "epfo"}

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly", "is.gd", "rb.gy", "shorturl.at", "tiny.cc", "ow.ly",
              "buff.ly", "rebrand.ly", "t.ly", "bitly.com", "s.id", "v.gd", "shorte.st", "adf.ly", "clck.ru", "qr.ae"}
RISKY_TLDS = {"top", "xyz", "tk", "ml", "ga", "cf", "gq", "icu", "buzz", "click", "link", "rest", "work", "cam", "cyou",
              "sbs", "cfd", "monster", "bond", "loan", "win", "zip", "mov", "support", "live", "shop", "online", "site", "fun", "vip"}
FREE_HOSTS = {"000webhostapp.com", "weebly.com", "weeblysite.com", "wixsite.com", "firebaseapp.com", "web.app", "pages.dev",
              "glitch.me", "ngrok.io", "ngrok-free.app", "netlify.app", "vercel.app", "github.io", "blogspot.com",
              "sites.google.com", "godaddysites.com", "square.site", "webflow.io", "framer.app", "my.canva.site", "duckdns.org",
              "repl.co", "herokuapp.com", "azurewebsites.net", "workers.dev", "r2.dev", "ipfs.io", "mybluehost.me"}
LURE_WORDS = ["secure", "verify", "verification", "login", "signin", "logon", "update", "account", "kyc", "billing",
              "wallet", "unlock", "suspend", "support", "helpdesk", "refund", "reward", "bonus", "claim", "auth", "confirm", "recovery"]
# Two-part endings where the registered domain has three labels (example.co.in)
TWO_PART_SUFFIXES = {"co.in", "org.in", "net.in", "gov.in", "ac.in", "edu.in", "res.in", "co.uk", "org.uk", "ac.uk", "gov.uk",
                     "com.au", "net.au", "org.au", "co.jp", "com.br", "com.cn", "com.mx", "co.za", "com.sg", "co.nz", "com.tr", "com.my",
                     "com.hk", "com.tw", "co.kr", "com.ar", "co.id", "com.vn", "com.ua", "com.pl", "co.th", "com.pk", "com.bd", "com.eg",
                     "com.sa", "co.il", "com.ng", "com.ph", "com.co", "com.pe", "com.ve", "co.ve", "com.np", "com.lk"}

URL_RE = re.compile(r"(?i)\b((?:https?|hxxps?)://[^\s<>\"']+|(?:[a-z0-9-]+\.)+[a-z]{2,24}(?:/[^\s<>\"']*)?)")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# (id, weight, explanation)
CHECKS = {
    "LK-01": (0.6, "Link is on a public phishing blocklist"),
    "LK-02": (0.5, "Uses a brand's name on a website that brand does not own"),
    "LK-03": (0.45, "Web address uses look-alike letters from another alphabet"),
    "LK-04": (0.3, "Link goes to a raw IP address instead of a website name"),
    "LK-05": (0.2, "Website ending often used by scammers"),
    "LK-06": (0.2, "Free hosting address often used for throwaway scam pages"),
    "LK-07": (0.2, "Login or payment words inside the website name"),
    "LK-08": (0.1, "Link shortener hides where the link really goes"),
}


def host_of(url: str) -> str:
    u = re.sub(r"(?i)^(https?|hxxps?)://", "", url.strip())
    u = re.split(r"[/?#:]", u, maxsplit=1)[0]
    u = u.split("@")[-1]                                    # user@host tricks
    return u.strip(".").lower()


SECOND_LEVEL = re.compile(r"^(com|co|net|org|gov|ac|edu|or|ne|go)\.[a-z]{2}$")


def registered_domain(host: str) -> str:
    parts = host.split(".")
    if len(parts) >= 3 and (".".join(parts[-2:]) in TWO_PART_SUFFIXES or SECOND_LEVEL.match(".".join(parts[-2:]))):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _edit1(a: str, b: str) -> bool:
    """True if a and b differ by exactly one insert, delete or substitution (paypa1 vs paypal, amazom vs amazon)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l_ = (a, b) if len(a) < len(b) else (b, a)
    return any(l_[:i] + l_[i + 1:] == s for i in range(len(l_)))


@lru_cache(maxsize=1)
def _blocklist():
    if not BLOCKLIST.exists():
        return frozenset()
    with gzip.open(BLOCKLIST, "rt", encoding="utf-8", errors="ignore") as f:
        return frozenset(line.strip().lower() for line in f if line.strip() and not line.startswith("#"))


def blocklist_size():
    return len(_blocklist())


def brand_impersonated(host: str):
    """Return the brand a host pretends to be, or None. Brand names and one-letter look-alikes count."""
    reg = registered_domain(host)
    label_text = host.translate(LEET)
    tokens = re.split(r"[.\-_]", label_text)
    for brand, owned in BRANDS.items():
        if reg in owned or host in owned or any(host.endswith("." + o) for o in owned):
            continue
        name, _, ending = reg.partition(".")
        if name == brand and ending not in RISKY_TLDS and (len(ending) == 2 or ending in {"com", "net", "org"} or ending in TWO_PART_SUFFIXES or SECOND_LEVEL.match(ending)):
            continue   # the brand's own site in another country (google.de, amazon.co.jp)
        if brand in SHORT_BRANDS:
            hit = brand in tokens
        else:
            hit = brand in label_text.replace("-", "").replace(".", "") if len(brand) >= 6 else brand in tokens
        if not hit and len(brand) >= 6:
            hit = any(_edit1(t, brand) for t in tokens if len(t) >= len(brand))   # gooogle, amazom; not "mail" vs "gmail"
        if hit:
            return brand
    return None


def check_url(url: str, use_blocklist=True):
    """Analyse one link. Returns (host, list of (check id, weight, why))."""
    host = host_of(url)
    found = []
    if not host or "." not in host:
        return host, found
    reg = registered_domain(host)
    tld = host.rsplit(".", 1)[-1]
    if use_blocklist and (host in _blocklist() or reg in _blocklist()):
        found.append("LK-01")
    brand = brand_impersonated(host)
    if brand:
        found.append("LK-02")
    if "xn--" in host or re.search(r"[^\x00-\x7f]", host):
        found.append("LK-03")
    try:
        ipaddress.ip_address(host)
        found.append("LK-04")
    except ValueError:
        pass
    if tld in RISKY_TLDS:
        found.append("LK-05")
    if reg in FREE_HOSTS or any(host.endswith("." + f) for f in FREE_HOSTS):
        found.append("LK-06")
    sub_and_name = host[: -len(reg)] + reg.split(".")[0] if reg else host
    if any(w in sub_and_name for w in LURE_WORDS) and reg not in {o for v in BRANDS.values() for o in v}:
        found.append("LK-07")
    if reg in SHORTENERS or host in SHORTENERS:
        found.append("LK-08")
    out = [(cid, CHECKS[cid][0], CHECKS[cid][1] + (f" ({brand})" if cid == "LK-02" else "")) for cid in found]
    return host, out


def url_risk(findings):
    """Combined risk of one link from its findings, 0 to 1."""
    return min(sum(w for _, w, _ in findings), 1.0)


def find_links(text: str):
    links = []
    for m in URL_RE.finditer(str(text)):
        u = m.group(1).rstrip(".,;:!?)'\"")
        h = host_of(u)
        tld = h.rsplit(".", 1)[-1] if "." in h else ""
        # skip things like "e.g", "U.S" or "file.txt": a bare domain needs a real-looking ending
        if not re.match(r"(?i)^(https?|hxxps?)://", u) and (len(tld) < 2 or tld in {"txt", "pdf", "jpg", "png", "doc", "exe", "zip"}
                                                             and "/" not in u):
            continue
        if re.fullmatch(r"[\d.]+", h) and h.count(".") != 3:
            continue
        links.append(u)
    return links


def check_text(text: str, use_blocklist=True):
    """All links in a message with their findings: list of dicts {url, host, findings, risk}."""
    out = []
    for u in find_links(text):
        host, f = check_url(u, use_blocklist)
        out.append({"url": u, "host": host, "findings": f, "risk": url_risk(f)})
    return out
