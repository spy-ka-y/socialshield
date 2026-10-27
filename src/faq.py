"""Question bank for the "Ask SocialShield" page.

Each entry: (topic, question, extra search words, answer). Answers are written from the project's measured
results, so the page never invents numbers. A typed question is matched to the closest entry (TF-IDF similarity).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

FAQ = [
    ("Basics", "What is SocialShield?", "about project overview what does socialshield do what does it do purpose summary explain architecture design",
     "SocialShield finds **fake social media accounts** and **scam messages** (phishing, money requests, impersonation, "
     "spam, misinformation), explains in plain words *why* something is suspicious, and lets an analyst **block, "
     "quarantine or dismiss** each alert. It works like an intrusion detection system, but for social media."),
    ("Basics", "Why is this a network security project?", "network security ids intrusion detection relevance course",
     "Fake accounts are attack infrastructure, like the infected computers in a botnet. SocialShield detects them the "
     "way a network **intrusion detection system** does: **signature rules** for known attacks (like Snort rules) plus "
     "**machine learning** for unusual behaviour, followed by response and logging. The link checker and the text "
     "normaliser are also standard network-security ideas: blocklists and traffic normalisation."),
    ("Basics", "What can it detect?", "detect types attacks categories alerts what kinds",
     "**Accounts:** fake or genuine, and the type of attacker (spam bot, phisher, money scammer, impersonator, "
     "misinformation spreader, adult-content scammer).\n\n**Messages:** phishing, money scams, impersonation, spam, "
     "misinformation and adult scams.\n\n**Links:** fake brand websites (like `paypa1.com` or `netflix-billing.top`) and "
     "domains on a list of 392,151 real phishing sites."),
    ("Basics", "How do I use it?", "use app pages navigate demo how to start tour",
     "Start on **Check a message**: pick an example or paste any message and press Scan. Then open the **Live feed** "
     "and press Start to watch threats appear in real time. **Accounts** explains why an account looks fake, and the "
     "**Dashboard** shows every alert so you can block or dismiss it."),

    ("Basics", "Who built SocialShield?", "who made built team members authors students group",
     "A team of three B.Tech Cybersecurity students at MPSTME, for the Network Security course: **Pravara Jain** "
     "(K020), **Yash Mehta** (K035) and **Somya Soni** (K063). The code is public at github.com/spy-ka-y/socialshield."),
    ("Basics", "What do the alert types mean?", "phishing money scam spam misinformation impersonation adult meaning define what is",
     "* **Phishing:** fake login pages or forms that steal passwords and OTPs\n* **Money Scam:** requests for money or "
     "fake investment profits\n* **Impersonation:** pretending to be a brand, an official or someone you know\n* "
     "**Spam:** mass promotional messages and chain forwards\n* **Misinformation:** false claims written to be shared "
     "quickly\n* **Adult Scam:** sexual bait leading to scams or blackmail"),
    ("Training", "What is TF-IDF?", "tfidf tf idf term frequency inverse document frequency vectorizer text numbers",
     "**Term Frequency, Inverse Document Frequency.** It turns text into numbers a model can use. A word scores high "
     "when it appears often in *this* message but rarely across all messages. Common words like *the* score low; "
     "telling words like *verify*, *OTP* or *gift card* score high. We use both whole words and short pieces of words, "
     "so misspellings still match."),
    ("Training", "Which dataset did you use?", "data dataset kaggle source accounts posts csv",
     "The Kaggle dataset **Fake Profile and Post Detection on Social Media**: **5,000 accounts** (3,751 genuine, "
     "1,249 fake) and **131,284 posts**, each already labelled. For honest testing we also used **5,574 real SMS "
     "messages** (UCI SMS Spam Collection), **92 messages written by our team**, and **392,151 real phishing domains** "
     "(Phishing.Database). None of the test data was used for training."),
    ("Training", "How did you train the account model?", "train account model random forest features how learned fake account",
     "1. Each account is turned into **32 numbers**: account age, followers, following, posts per day, posting at "
     "night, repeated posts, links per post and so on.\n2. The data is split: **4,000 accounts** to learn from, "
     "**1,000** kept aside for testing.\n3. A **Random Forest** (300 decision trees that each ask yes/no questions and "
     "then vote) learns which patterns mean *fake*.\n4. It is tested on the 1,000 accounts it never saw."),
    ("Training", "How did you train the message model?", "message model tfidf logistic regression text words train",
     "**TF-IDF** turns each message into numbers: words that are frequent in this message but rare overall score high, "
     "so words like *verify*, *OTP* or *send money* stand out. A **Logistic Regression** model then learns which words "
     "go with which scam type. It trained on posts from 80% of the accounts and was tested on posts from the other 20%, "
     "so no account appears in both."),
    ("Training", "What tools and libraries did you use?", "tools libraries python scikit learn pandas streamlit tech stack software",
     "**Python**, **pandas** for data, **scikit-learn** for the models, **Plotly** and **NetworkX** for charts and the "
     "spread network, **Streamlit** for this app, and **GitHub** plus **Streamlit Community Cloud** for hosting. "
     "An optional local AI uses **Ollama** with the Qwen 2.5 model."),
    ("Training", "Why Random Forest?", "why random forest choice svm logistic regression compare models",
     "We compared Logistic Regression, SVM and Random Forest; all three scored 100% on this dataset. Random Forest was "
     "chosen because it also says **which features mattered**, so the app can explain every flagged account (for "
     "example *account age 15 days, typical genuine account 874 days*)."),
    ("Training", "Which features matter most?", "important features clues signals top features importance",
     "The strongest clues are **account age**, **average likes, comments and shares**, **average post length**, "
     "**followers and following**, and **posting at night**. A typical fake account is about 39 days old against 874 "
     "for a genuine one, and gets around 4 likes per post against about 32."),

    ("Results", "How accurate is it?", "accuracy results score performance percent how good",
     "On the Kaggle test data: **100%** for accounts and messages. That number is real but misleading, because the "
     "dataset is synthetic. On harder tests:\n\n* **30 unseen team-written messages:** 50% of attacks caught, 17% false "
     "alarms (56% with the link checker)\n* **5,574 real SMS:** 23% of spam caught, only 0.3% false alarms\n* **Same "
     "design retrained on real SMS:** 95% caught, 0.3% false alarms"),
    ("Results", "Why is the accuracy 100%? Isn't that suspicious?", "100 percent suspicious too good overfitting synthetic perfect",
     "Yes, and we say so. The dataset is **synthetic** (computer-generated): fake accounts are on average 39 days old, "
     "genuine ones 874, so they barely overlap and any model separates them. That is why we tested on real SMS, on "
     "messages we wrote ourselves, and with attackers who imitate real users. Those tests show the real picture."),
    ("Results", "How does it do on real messages?", "real world real data sms uci transfer generalise",
     "On **5,574 real SMS messages** it caught only **23%** of spam, but when it raised an alert it was right **92%** "
     "of the time (0.3% false alarms). It was trained on a different style of message. Retraining the same design on "
     "real SMS raised detection to **95%**, and only **200 labelled real messages** already reach about 89%. Lesson: "
     "a model only knows the data it learned from."),
    ("Results", "What are the limitations?", "limitations weaknesses problems drawbacks fails",
     "* Trained on a **synthetic** dataset, so it transfers poorly to real messages (23% on real SMS).\n* The DMs, the "
     "spread paths and the Block/Quarantine actions are **simulated**; it is not connected to a real platform.\n* Fake "
     "accounts that copy genuine behaviour beyond about 40% are mostly missed.\n* The local AI flagged two thirds of "
     "genuine messages, so it is off by default.\n* Only English (and Hinglish in English letters) was tested."),

    ("How it works", "How does it decide if a message is a scam?", "decide risk score combine layers how works message scam",
     "1. **Normaliser:** undoes hidden tricks (invisible characters, look-alike letters, `0TP`, `p a y`).\n2. **37 "
     "signature rules** look for known scam patterns such as *share your OTP* or *pay a processing fee*.\n3. **Link "
     "checker** inspects every web address.\n4. **ML model** gives its probability for each scam type.\n\nThese are "
     "added into **one risk score**. An alert is raised at 50% or more, with the matched words highlighted."),
    ("How it works", "How does the link checker work?", "link links url urls domain domains phishing blocklist website websites check checked checking address",
     "Every web address is checked against a blocklist of **392,151 real phishing domains**, and for warning signs in "
     "the address itself: a brand name on a site the brand does not own (`sbi-kyc-update.xyz`), one-letter look-alikes "
     "(`paypa1`), letters from another alphabet, raw IP addresses, risky endings like `.top`, free hosting, login words "
     "and link shorteners. Without the blocklist, these checks alone flagged **50% of 10,000 real phishing domains** "
     "and only **0.7%** of the 10,000 most popular real websites."),
    ("How it works", "Can scammers trick it?", "evasion fool trick bypass hackers hide invisible characters attack evade defeat",
     "Yes, and we tested how. Hiding words with **invisible characters** cut detection of our scam messages to 9%, and "
     "**look-alike Cyrillic letters** to 18%. Our **normaliser** undoes these tricks before anything reads the message "
     "and brings detection back to **100%**, without a single new false alarm. Fake accounts that copy real users are "
     "harder: at 40% imitation only about half are caught."),
    ("How it works", "What happens when a threat is found?", "alert respond block quarantine incident log response",
     "An alert appears with the scam type, risk level, the highlighted words, what the attacker wants and what to do. "
     "The analyst chooses **Block** (stop the sender), **Quarantine** (hide the message) or **Not a threat** (false "
     "alarm). Every alert and action is saved in the **incident log** and can be exported: detect, respond, record, "
     "like a security operations centre."),
    ("How it works", "How does misinformation spread?", "misinformation spread network shares viral fake news",
     "In the dataset, misinformation posts get **93.5 shares per 100 followers**, against 0.6 to 3.5 for every genuine "
     "account type, so they spread far beyond their author's audience. The share counts are real; the paths of who "
     "shared from whom are simulated, because the dataset does not record them."),
    ("How it works", "What does the local AI do?", "ai llm ollama qwen local ai second opinion chatgpt",
     "An optional AI (Qwen 2.5, running on the laptop with Ollama, so no message leaves the device) gives a second "
     "opinion. It caught 94% of unseen attacks but also flagged **67% of genuine messages**, mostly anything mentioning "
     "money. So it is **off by default**, and by design it can add an alert but never remove one, because an attacker "
     "could try to talk an AI into saying *safe* (prompt injection)."),
    ("How it works", "What is real and what is simulated?", "real simulated fake data honest what is real",
     "**Real:** the trained models, the share counts, the 5,574 real SMS, the 392,151 real phishing domains, and the "
     "92 messages our team wrote. **Simulated:** the dataset itself (synthetic), DMs as a channel, who-shared-from-whom "
     "in the spread network, and the effect of Block and Quarantine."),

    ("Future", "How could the accuracy be improved?", "improve improvement accuracy better future work increase next steps",
     "1. **Train on real labelled messages from Indian platforms.** Our learning curve shows even a few hundred real "
     "examples lift detection from 23% to about 89%.\n2. **Use a multilingual language model** (for example a "
     "fine-tuned multilingual BERT) to understand meaning and Hindi or Hinglish, not just keywords.\n3. **Learn from "
     "analysts:** every *Not a threat* click becomes a new training example.\n4. **Add signals scammers cannot fake "
     "cheaply:** device fingerprints, IP reputation, and how accounts connect to each other.\n5. **Check where links "
     "really lead:** follow redirects and look at domain age, not just the address."),
]


class FAQIndex:
    def __init__(self):
        docs = [f"{q} {q} {kw}" for _, q, kw, _ in FAQ]
        # whole words plus pieces of words, so "links", "checked" or small typos still match
        self.vec = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
                                 ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])
        self.m = self.vec.fit_transform(docs)

    def best(self, text):
        """(entry, similarity) for the closest prepared question."""
        sims = cosine_similarity(self.vec.transform([text]), self.m)[0]
        i = int(sims.argmax())
        return FAQ[i], float(sims[i])
