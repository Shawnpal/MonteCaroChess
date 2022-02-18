"""
Starter Code for Assignment 1 - COMP 8085
Please do not redistribute this code our your solutions
This is the frontend web app created for you to test your implementation.
"""
# pip install flask

from flask import Flask, render_template, request, jsonify
from engine import AIEngine
app = Flask("COMP8085_Assignment1", static_url_path='', static_folder='static')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/call_board', methods=['POST'])
def call_board():
    game_engine = AIEngine(request.form['fen'])
    game = game_engine.game
    ai = game_engine.computer
    if game.status < 2:
        ai_algebraic_move = ai.make_move(str(game))
        game.apply_move(ai_algebraic_move)
        return jsonify(move=ai_algebraic_move, fen=str(game))
    return "Game Over"


if __name__ == "__main__":
    port = 3636
    app.run(host="localhost", port=port)
