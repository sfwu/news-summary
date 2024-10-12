from gevent import monkey
from gevent import pywsgi
from flask import Flask, request, render_template
import argparse
from summarizer import summarize, load_fine_tuned_model
import torch
import os

monkey.patch_all()

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--http_id', type=str, default="0.0.0.0", help='server ip')
    parser.add_argument('--port', type=int, default=5555, help='serve port')
    return parser.parse_args()


def start_sever():
    args = set_args()
    app = Flask(__name__)
    model,tokenizer = load_fine_tuned_model()

    @app.route('/')
    def index():
        return "This is a news summarizer"

    @app.route('/news-summarizer', methods=['Get', 'POST'])
    def response_request():
        if request.method == 'GET':
            return render_template("index.html")

        news_text = request.form.get('news_text')
        temp_max_length = request.form.get('max_length')
        temp_top_k = request.form.get('top_k')
        temp_top_p = request.form.get('top_p')
        max_length = int(temp_max_length) if isinstance(temp_max_length, str) and temp_max_length.isdigit() else 250
        top_k = int(temp_top_k) if isinstance(temp_top_k, str) and temp_top_k.isdigit() else 5
        top_p = int(temp_top_p) if isinstance(temp_top_p, str) and temp_top_p.isdigit() else 0.7
        summary = summarize(model=model, tokenizer=tokenizer, article=news_text, max_length=max_length, top_k=top_k, top_p=top_p)
        return render_template("post_index.html",
                                news_text=news_text,
                                summary=summary,
                                max_length=max_length,
                                top_k=top_k,
                                top_p=top_p)

    server = pywsgi.WSGIServer((str(args.http_id), args.port), app)
    server.serve_forever()

if __name__ == '__main__':
    start_sever()