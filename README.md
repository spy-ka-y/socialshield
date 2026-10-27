# SocialShield

ML-based Fake Account and Social Engineering Attack Detection. Network Security project, MPSTME.

SocialShield detects fake and bot accounts, flags malicious posts and DMs (phishing, money scams,
spam, misinformation, impersonation, adult scams), lets an analyst respond to every alert
(block, quarantine, false alarm), shows how misinformation spreads, and tests how well the
models hold up when attackers adapt.

## The app (6 pages)
| Page | What it does |
|---|---|
| Dashboard | What SocialShield does in 5 steps, headline numbers, open alerts (Block / Quarantine / Not a threat), who is attacking, incident log |
| Check a message | Scan any message with a full explanation; DM inbox of team-written or real messages |
| Live feed | Posts or real SMS stream in and are scanned live |
| Fake accounts | Why an account looks fake, and what kind of attacker it is |
| Research | Misinformation spread, evasion tests, real-world check |
| Ask SocialShield | Ask questions about the project (prepared answers from measured results) or paste a message to check |

## Detection layers
0. **Normaliser**: undoes evasion tricks (invisible characters, look-alike letters, leetspeak,
   spaced or dotted letters, disguised links) before anything else reads the message,
   like traffic normalisation in an IDS. Hiding words is itself treated as a warning sign.
1. **Signature rules**: 37 known scam patterns (like antivirus or IDS signatures).
2. **Link checker**: every web address in a message is checked for brand look-alikes
   (paypa1, g00gle, netflix-billing.top), look-alike letters, raw IP addresses, risky endings,
   free hosting, login words and link shorteners, and against a blocklist of 392,151 real
   phishing domains (Phishing.Database).
3. **Message ML model**: TF-IDF + Logistic Regression trained on 131,284 posts.
4. **Account model**: Random Forest on profile and behaviour features, plus an attack-type model.
5. **Optional local AI second opinion**: a small language model run locally with Ollama
   (default `qwen2.5:3b`). It can add an alert the other layers missed but can never clear one,
   and messages that try to instruct the AI (prompt injection) are treated as attacks.

## Run locally (Windows)
Double-click `SocialShield.bat`. It installs everything on first run, starts Ollama in the
background, downloads and preloads the AI model if needed, and opens the app in your browser.
(Without Ollama installed it still works, just without the AI second opinion.)

## Run anywhere else
```
pip install -r requirements.txt
streamlit run app.py
```
On first start the app trains its models automatically (about 2 minutes).
The AI layer is skipped when Ollama is not available, for example on Streamlit Cloud.

## How the models are trained (Colab)
Open [notebooks/SocialShield_Training.ipynb](https://colab.research.google.com/github/spy-ka-y/socialshield/blob/main/notebooks/SocialShield_Training.ipynb) in Google Colab and choose Runtime > Run all. It walks through every training step with charts, in about 3 minutes.

## Research experiments
| Script | What it measures |
|---|---|
| `python src/evaluate.py` | Feature removal and account evasion (fake accounts imitating real users) |
| `python src/evaluate_ai.py` | Rules + ML + local AI on the team-written DM tests (needs Ollama) |
| `python src/real_data_test.py` | The detector on 5,574 REAL SMS messages, retraining and a learning curve |
| `python src/message_evasion_test.py` | Obfuscation attacks on messages, with and without the normaliser |
| `python src/link_test.py <phishing_domains.txt> <top_domains.txt>` | Link checks on 10,000 real phishing domains and the 10,000 most popular real sites |

Results are saved in `reports/` and shown on the app's Research page. The Live feed and the DM inbox can also switch
from the synthetic dataset to the 5,574 real SMS messages, with each message's true label shown.

## Change the colour theme
`python src/set_theme.py light` (or `midnight`, `terminal`, `slate`), then restart the app.

## Data
- `data/raw_user_profiles.csv`, `data/raw_user_activities.csv`: Kaggle "Fake Profile and Post Detection on Social Media" (synthetic).
- `data/dm_messages.csv` (Test 1) and `data/dm_test2.csv` (Test 2, unseen): team-written DM test sets (`text,label`). Used only for testing, never for training.
- `data/external/phishing_domains.txt.gz`: active phishing domains from Phishing.Database (github.com/mitchellkrogza/Phishing.Database), downloaded 27 Oct 2026, with 13 popular legitimate sites removed.
- `data/external/sms.tsv`: UCI SMS Spam Collection (Almeida, Gomez Hidalgo and Yamakami, 2011; CC BY 4.0), real messages used only for the real-world check.
- DMs and spread paths are simulated because the dataset has neither. Share counts are real.
- `account_type` and `content_type` are never used as features because they reveal the label.
