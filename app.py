# import flask for get/post request handling and sqlite for database usage
from flask import request
from flask import Flask, render_template, g, session, url_for
import sqlite3
# import translate function from ml file 
from translator import runTranslate

# create instance of flask object
app = Flask(__name__)

# opening the page for the first time will lead to the root directory and a get request will be received 
@app.route('/', methods=["GET"])
def index():
    # set all variables to default including input and output language in the db
    input=''
    output=''
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute("UPDATE language SET inputLang = ? WHERE id = 1", ('Input language',))
    con.commit()
    cur.execute("UPDATE language SET outputLang = ? WHERE id = 1", ('Output language',))
    con.commit()
    # commit changes to db and return render template function to load the html page with all the referenced variables set to default 
    return render_template('frontend.html', input=input, output=output, inputLang='Input language', outputLang='Output language')

# clicking any of the languages on the dropdown menu or clicking the translate button will cause a post request to be sent to the server and the user will be routed to /translate
@app.route('/translate', methods=["POST"])
def translate():
    # connect to database and create a cursor
    con = sqlite3.connect('database.db')
    cur = con.cursor()
    # take user input from input textarea
    text = request.form['inputText']
    new = ''
    # get input and output languages from respective dropdowns 
    inputLang = request.form.get('inputLang')
    outputLang = request.form.get('outputLang')
    # update the input and output language in the database so when another post request is sent the selected information will be saved 
    if inputLang != None:
        cur.execute("UPDATE language SET inputLang = ? WHERE id = 1", (inputLang,))
        con.commit()
    if outputLang != None:
        cur.execute("UPDATE language SET outputLang = ? WHERE id = 1", (outputLang,))
        con.commit()
    # if neither input language or output language are selected (i.e. the user clicked the Translate button) fetch the input and output languages
    if inputLang == None and outputLang == None:
        cur.execute("SELECT * FROM language")
        inputl = cur.fetchall()[0][1]
        cur.execute("SELECT * FROM language")
        outputl = cur.fetchall()[0][2]
        # if either of the dropdown languages are unselected return an error
        if inputl == 'Input language' or outputl == 'Output language':
            return ('Please select both an input and output language!', 400)
        else:
            if len(text) == 0:
                new = ""
            else:
                if inputl == outputl:
                    new = text
                else:
                    # if there is an input and output language selected, the text box is not empty, and the input and output languages are different, call the translate function on the respective languages
                    if outputl == 'Japanese':
                        new = runTranslate(text, "ja")
                    elif outputl == 'English':
                        new = runTranslate(text, "en")
                
    # fetch current input and output language from database
    cur.execute('SELECT * FROM language')
    currentInputLang = cur.fetchall()[0][1]
    cur.execute('SELECT * FROM language')
    currentOutputLang = cur.fetchall()[0][2]
    # return html page with new output text and new selected languages
    return render_template('frontend.html', input=text, output=new, inputLang=currentInputLang, outputLang=currentOutputLang)
