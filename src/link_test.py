"""Measure the link checker on REAL domains.

Phishing: a random sample of domains from Phishing.Database (active phishing domains, public feed).
Legitimate: the 10,000 most popular domains from a public popularity list (zer0h/top-1000000-domains).
The blocklist check (LK-01) is switched OFF here, otherwise every phishing domain would be found by lookup;
this measures only the heuristics, which are what catch brand-new phishing domains no list has seen.

Run from the project root:  python src/link_test.py <phishing_domains.txt> <top_domains.txt>
Writes reports/link_results.json
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from link_checker import CHECKS, check_url, registered_domain, url_risk  # noqa: E402

SEED, N_PHISH = 42, 10000
THRESHOLDS = [0.2, 0.3, 0.45]


def load(path):
    return [l.strip().lower() for l in open(path, encoding="utf-8", errors="ignore") if l.strip() and not l.startswith("#")]


def main():
    phish_all, legit = load(sys.argv[1]), load(sys.argv[2])
    legit_set = set(legit)
    phish_all = [d for d in phish_all if d not in legit_set and registered_domain(d) not in legit_set]
    random.Random(SEED).shuffle(phish_all)
    phish = phish_all[:N_PHISH]
    phish_set = set(phish_all)
    legit = [d for d in legit if d not in phish_set]

    def score(domains):
        rows = []
        for d in domains:
            _, f = check_url(d, use_blocklist=False)
            rows.append((d, [c for c, _, _ in f], url_risk(f)))
        return rows
    P, L = score(phish), score(legit)
    out = {"n_phishing": len(P), "n_legit": len(L), "phishing_pool": len(phish_all), "thresholds": {}}
    for t in THRESHOLDS:
        out["thresholds"][str(t)] = {
            "phishing_caught": round(sum(r >= t for _, _, r in P) / len(P), 4),
            "legit_flagged": round(sum(r >= t for _, _, r in L) / len(L), 4),
        }
        print(f"threshold {t}: phishing caught {out['thresholds'][str(t)]['phishing_caught']:.1%}, "
              f"legit flagged {out['thresholds'][str(t)]['legit_flagged']:.1%}")
    cp, cl = Counter(c for _, cs, _ in P for c in cs), Counter(c for _, cs, _ in L for c in cs)
    out["per_check"] = {c: {"why": CHECKS[c][1], "phishing": round(cp[c] / len(P), 4), "legit": round(cl[c] / len(L), 4)}
                        for c in CHECKS if c != "LK-01"}
    for c, v in out["per_check"].items():
        print(f"  {c} {v['why'][:55]:55s} phishing {v['phishing']:6.1%}  legit {v['legit']:6.1%}")
    out["legit_flagged_examples"] = [(d, cs) for d, cs, r in L if r >= 0.3][:15]
    out["phishing_missed_examples"] = [d for d, cs, r in P if r < 0.2][:15]
    out["phishing_caught_examples"] = [(d, cs) for d, cs, r in P if r >= 0.45][:15]
    (ROOT / "reports/link_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved reports/link_results.json")


if __name__ == "__main__":
    main()
