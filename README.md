# Instagram Chat Analyzer

A robust **Streamlit** app to analyze your **Instagram Direct Messages** with clean UI and advanced analytics:

- Load data via ZIP, multiple JSON files, or local folder (best for GB-sized exports)
- Clean and normalize text to remove encoding artifacts (mojibake)
- Filter out system messages, support English + Hinglish stopwords
- View top words, bigrams, emojis, response times, activity heatmaps
- Detect conversation patterns: daily starters, longest gaps, streaks
- Search messages by sender or text
- Export results as CSV and summary JSON

---

##  Repository

**Repo:** https://github.com/kart1kxd/instagram-chat-analyzer

---

## 📥  Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/kart1kxd/instagram-chat-analyzer.git
   cd instagram-chat-analyzer
   ```

2. **Create a Python virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: `venv\Scripts\activate`
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Navigate to `http://localhost:8501`.

---

##  How to Download Instagram Data

1. Head to Instagram settings → **Privacy and Security** → **Data Download**
2. Request a **JSON** format download
3. Download the ZIP sent via email
4. Unzip locally — you should see:

   ```
   messages/
     inbox/
       <chat-1>/
         message_1.json
         message_2.json
       <chat-2>/
       ...
   ```

---

##  Loading Options within the App

- **Upload ZIP** — works for smaller exports (<200 MB)
- **Upload multiple JSONs** — select multiple `message_*.json` files
- **Local folder path** — ideal for large exports, no upload required (run locally)

---

##  How to Use the App

- Pick a conversation from the dropdown
- Switch between tabs:
  - **Overview**: message counts, daily starters, longest gaps, streaks
  - **Words**: top words/bigrams and who uses them
  - **Emojis**: emoji usage stats
  - **Timing**: activity trends and response times
  - **Media**: counts of photos/videos/shares/stickers
  - **Search**: filter messages by sender or content
  - **Export**: download CSVs and summary JSON

- **Sidebar settings** for:
  - Date range
  - Hinglish/custom stopwords
  - System token filtering
  - Participant aliases
  - N-gram toggle (bigrams)

---

##  Export Options

Within the **Export** tab:
- **Messages CSV** — conversation messages
- **Words CSV** — word usage stats
- **Summary JSON** — includes stats, streaks, gaps, date range, participants

---

##  Development & Deployment

- For local development: `streamlit run app.py`
- For deployment, use:
  - `requirements.txt`
  - `start.sh` or `.bat`
  - Optionally: `Procfile` (Heroku), `Dockerfile` (Docker/Render)

---

##  Contributions

Got ideas or improvements? Pull requests and issue submissions are welcome!

---

Made with Python, Streamlit, and ❤️  
Supports both English and Hinglish DM analysis.
