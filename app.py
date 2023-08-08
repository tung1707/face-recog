from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
from routes.route import recognition
 
app = Flask(__name__)
 
app.register_blueprint(recognition) 
 
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
