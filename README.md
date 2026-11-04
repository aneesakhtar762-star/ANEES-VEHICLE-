# 🚗 Vehicle Info Finder

A Streamlit app where you enter a vehicle's **make and model** and instantly see its **fuel average**, specifications and features. Powered by the [Groq API](https://console.groq.com).

## Features
- Enter make, model, and optionally year, variant/engine and country
- **Fuel economy** (city / highway / combined) shown in **km/l, L/100 km or mpg**
- Engine & drivetrain, dimensions, performance, and feature lists (safety, comfort, infotainment, exterior)
- Pros & cons, plus support for **electric vehicles** (battery and range)
- **Monthly fuel cost estimator** from your distance and fuel price
- **Data confidence** rating (High / Medium / Low) and a warning when the AI is unsure
- Download the result as a Markdown report

## Important: accuracy
Details come from an AI model's general knowledge, **not a live vehicle database**. Numbers can be off or differ by variant, year and market. The app blanks out values the AI is unsure about (shown as `N/A`), but always **verify with the manufacturer or an authorised dealer** before making a buying decision.

## Project structure
```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. Get a free API key at <https://console.groq.com/keys>.
2. Install:
   ```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Provide your key (pick one):
   - Create `.streamlit/secrets.toml`:
     ```toml
     GROQ_API_KEY = "gsk_your_key_here"
     ```
   - or set an environment variable: `export GROQ_API_KEY=gsk_your_key_here` (Windows: `set GROQ_API_KEY=...`)
   - or paste the key into the app's sidebar.
4. Run:
   ```bash
   streamlit run app.py
   ```

## Push to GitHub (using the website, no command line)

1. Go to <https://github.com> and sign in.
2. Click **+** (top right) → **New repository**.
3. Name it (e.g. `vehicle-info-finder`), choose Public or Private, click **Create repository**.
4. On the empty repo page click **uploading an existing file**.
5. Drag in `app.py`, `requirements.txt` and `README.md`.
   - Names must be exact: `requirements.txt` (with an **s**), in the repo root, not in a folder.
6. Type a commit message (e.g. `Initial commit`) and click **Commit changes**.
7. **Protect your API key.** Never upload a file containing it. Add a `.gitignore` via *Add file → Create new file*:
   ```
   .streamlit/secrets.toml
   venv/
   __pycache__/
   .env
   ```

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and **Continue with GitHub**; authorise access.
2. Click **Create app** → **Deploy a public app from GitHub** (for a private repo, grant access when asked).
3. Fill in:
   - **Repository:** `your-username/vehicle-info-finder`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings** → **Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**. After a few minutes you get a URL like `https://your-app-name.streamlit.app`.

**Update later:** open a file on GitHub → pencil icon → **Commit changes**; Streamlit redeploys automatically.
**Change the key:** Streamlit Cloud → app **⋮ → Settings → Secrets**.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` on deploy | The file must be named `requirements.txt` and sit in the repo root |
| "Groq rejected the API key" | Re-copy the key into Secrets; no stray spaces |
| "Could not recognise that vehicle" | Check spelling; try the common name (e.g. `Suzuki` + `Alto`) |
| Many `N/A` values | The AI wasn't sure. Add the year, variant and country, or check the manufacturer's site |
| Rate-limit error | Wait a minute or switch model in the sidebar |
| Model not found | Groq retires models sometimes. Edit the `MODELS` list at the top of `app.py` ([model list](https://console.groq.com/docs/models)) |
