from flask import Flask, request, render_template_string
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Enhanced CSS with a focus on "Missing Keywords" tags and a Score Dashboard
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TalentScan AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --brand: #6366f1;
            --brand-gradient: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
            --bg: #f3f4f6;
            --card: #ffffff;
            --text-main: #111827;
            --text-muted: #6b7280;
        }

        body { 
            font-family: 'Plus Jakarta Sans', sans-serif; 
            background-color: var(--bg); 
            color: var(--text-main);
            margin: 0; padding: 40px 20px;
            display: flex; justify-content: center;
        }

        .container {
            width: 100%; max-width: 800px;
            background: var(--card);
            padding: 40px; border-radius: 24px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }

        header { text-align: center; margin-bottom: 40px; }
        h1 { font-weight: 800; font-size: 2.5rem; margin: 0; color: var(--brand); letter-spacing: -1px; }
        p.subtitle { color: var(--text-muted); margin-top: 8px; }

        .form-section { display: grid; gap: 24px; }
        
        label { font-weight: 600; font-size: 0.95rem; display: block; margin-bottom: 8px; }

        input[type="file"] {
            width: 100%; padding: 12px; border: 2px dashed #e5e7eb;
            border-radius: 12px; background: #f9fafb; cursor: pointer; box-sizing: border-box;
        }

        textarea {
            width: 100%; border: 1px solid #e5e7eb; border-radius: 12px;
            padding: 16px; font-family: inherit; font-size: 1rem;
            box-sizing: border-box; min-height: 180px;
        }

        button {
            background: var(--brand-gradient); color: white; border: none;
            padding: 16px; border-radius: 12px; font-weight: 700; font-size: 1rem;
            cursor: pointer; transition: all 0.2s ease;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4); }

        /* Results Styling */
        .results-container {
            margin-top: 40px; padding-top: 40px; border-top: 1px solid #f3f4f6;
            animation: fadeIn 0.5s ease-out;
        }

        .score-box {
            text-align: center; margin-bottom: 30px;
        }
        .score-circle {
            font-size: 3.5rem; font-weight: 800; color: var(--brand);
            display: block; margin-bottom: 5px;
        }

        .keyword-title { font-weight: 700; margin-bottom: 15px; font-size: 1.1rem; }
        .keyword-list { 
            display: flex; flex-wrap: wrap; gap: 10px; padding: 0; list-style: none;
        }
        .keyword-tag {
            background: #fee2e2; color: #b91c1c; padding: 6px 14px;
            border-radius: 20px; font-size: 0.85rem; font-weight: 600;
        }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>TalentScan AI</h1>
        <p class="subtitle">Intelligent Resume-to-Job Matching</p>
    </header>

    <form method="post" enctype="multipart/form-data" class="form-section">
        <div>
            <label>Candidate Resume (PDF)</label>
            <input type="file" name="resume" accept=".pdf" required>
        </div>

        <div>
            <label>Job Description</label>
            <textarea name="jd" placeholder="Paste the job requirements here..." required></textarea>
        </div>

        <button type="submit">Analyze Candidate</button>
    </form>

    {% if score %}
    <div class="results-container">
        <div class="score-box">
            <span class="score-circle">{{score}}%</span>
            <span style="color: var(--text-muted); font-weight: 600;">Match Accuracy</span>
        </div>

        <div class="keywords-section">
            <div class="keyword-title">Top Missing Keywords</div>
            <ul class="keyword-list">
                {% for word in missing %}
                <li class="keyword-tag">{{word}}</li>
                {% endfor %}
            </ul>
        </div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

def extract_text(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except:
        pass
    return text

def analyze(resume, jd):
    # Standardizing text for better comparison
    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(matrix)[0][1] * 100

    # Improved keyword extraction logic
    resume_words = set(resume.lower().replace(',', '').replace('.', '').split())
    jd_words = set(jd.lower().replace(',', '').replace('.', '').split())
    
    # Filter out very short words (noise)
    missing = [word for word in (jd_words - resume_words) if len(word) > 3]

    return round(score, 2), missing[:12]

@app.route("/", methods=["GET","POST"])
def index():
    score = None
    missing = []
    if request.method == "POST":
        file = request.files.get("resume")
        jd = request.form.get("jd")
        if file and jd:
            resume_text = extract_text(file)
            score, missing = analyze(resume_text, jd)
    return render_template_string(HTML, score=score, missing=missing)

if __name__ == "__main__":
    app.run(debug=True)