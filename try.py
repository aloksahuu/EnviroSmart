from flask import Flask, render_template
from flask_jsglue import JSGlue  # Import JSGlue

app = Flask(__name__)
jsglue = JSGlue(app)  # Initialize JSGlue

@app.route('/')
def home():
    return render_template('base.html')

if __name__ == '__main__':
    app.run(debug=True)

