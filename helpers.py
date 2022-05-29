from flask import render_template

def apology(message, code):

    # Render message as an apology to user, pass error code and message
    return render_template("apology.html", code=code, message=message)