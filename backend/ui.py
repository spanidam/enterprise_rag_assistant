from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from answer_engine import answer_question

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body>
        <h2>Enterprise RAG Assistant</h2>
        <form method="post">
            <textarea name="question" rows="4" cols="60"></textarea><br><br>
            <button type="submit">Ask</button>
        </form>
    </body>
    </html>
    """

@app.post("/", response_class=HTMLResponse)
def ask(question: str = Form(...)):
    answer = answer_question(question)
    return f"""
    <html>
    <body>
        <h3>Question</h3>
        <p>{question}</p>
        <h3>Answer</h3>
        <p>{answer}</p>
        <a href="/">Ask another</a>
    </body>
    </html>
    """