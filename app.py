import pyodbc
import nltk
import sys
import string
import torch
import random
import numpy as np
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from flask import Flask, json, request, render_template
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
app = Flask(__name__)
lem = WordNetLemmatizer()

## LOAD AI MODELS ##
#model = AutoModelForSequenceClassification.from_pretrained("pytorch_models/model/roberta-base")
#tokenizer = AutoTokenizer.from_pretrained("roberta-base")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.load_state_dict(torch.load("pytorch_models/roberta_boolQ/roberta_boolQ_weights.pth", map_location=lambda storage, loc: storage))

## CONNECT TO DATABASE ##
conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=C:/xampp/htdocs/agi_ticker_tape_new/database.accdb;')
cursor = conn.cursor()

## ##
FILE_UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

nouns_pos = ["NN", "NNP", "NNS", "NNPS", "PRP", "PRP$", "WP", "WP$", "JJ", "JJR", "JJS"]
verbs_pos = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "RB", "RBR", "RBS", "WRB"]

## DISPLAY HTML PAGES ##
@app.route("/query", methods=["GET", "POST"])
def display_query():
    return render_template("query.html")

@app.route("/management", methods=["GET", "POST"])
def display_management():
    return render_template("management.html")

@app.route("/diagnostics", methods=["GET", "POST"])
def display_diagnostics():
    return render_template("diagnostics.html")

@app.route("/about", methods=["GET", "POST"])
def display_about():
    return render_template("about.html")

@app.route("/upload_file", methods=["GET", "POST"])
def parse_uploaded_file():
    file = request.files['file_input']
    file.save("temporary_files/" + file.filename)
    parse_new_knowledge("temporary_files/" + file.filename)
    return json.dumps({"response": file.filename + " was Successfully Uploaded"})


@app.route("/scrape_website", methods=["GET", "POST"])
def scrape_website():
    website = str(request.get_data()).split("'")[1]
    html_page = requests.get(website).content
    soup = BeautifulSoup(html_page, 'html.parser')
    text_list = re.sub(re.compile('<.*?>'), '', str(soup.find_all('p')))
    file = open("temporary_files/website_input.txt", "w")
    file.write(text_list)
    file.close()
    parse_new_knowledge("temporary_files/website_input.txt")

    return json.dumps({"response": website + " was Successfully Scraped"})

## HELPER FUNCTIONS TO RETURN JSON TO HTML PAGES ##
@app.route("/return_table", methods=["GET", "POST"])
def return_table():
    selected_table = str(request.get_data()).split("'")[1]
    cursor.execute("SELECT * FROM " + selected_table)
    records = cursor.fetchall()
    html_string = '''
    <thead>
      <tr>'''

    for item in cursor.columns(table=selected_table):
        html_string += "<th>" + str(item[3]) + "</th>"

    html_string += '''
    </tr>
    </thead>
    <tbody>'''

    for item in records:
        html_string += "<tr>"
        for value in item:
            html_string += "<td>" + str(value) + "</td>"
        html_string += "</tr>"

    html_string += "</tbody>"
    return json.dumps({"table": html_string})

#@app.route("/query_response", methods=["GET", "POST"])
#def get_query_response():
#    question = str(request.get_data()).split("'")[1]
#    context = []
#    nouns = []
#    for word in nltk.pos_tag(nltk.word_tokenize(question)):
#        if(word[1] in nouns_pos):
#            nouns.append(word[0])

#    word_id = []
#    for word in nouns:
#        cursor.execute("SELECT * FROM Words WHERE Word = ?", word)
#        word = cursor.fetchone()
#        if word:
#            word_id.append(str(word[0]))

#    cursor.execute("SELECT * FROM Statements")
#    for record in cursor.fetchall():
#        word_missing = False
#        for id in word_id:
#            if (id in record[1]):
#                if(record[3] not in context):
#                    context.append(record[3])

#    passage = ""
#    for item in context:
#        passage += item + " "

    #print(passage + "\n")
    #print(passage[0:512] + "\n")

#    sequence_blank = tokenizer.encode(passage, max_length=512, truncation=True)
#    sequence = tokenizer.encode_plus(question, passage, return_tensors="pt", max_length=512, truncation=True)['input_ids'].to(device)
#    print(tokenizer.decode(sequence_blank))
#    logits = model(sequence)[0]
#    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
#    proba_yes = round(probabilities[1], 2)
#    proba_no = round(probabilities[0], 2)

#    answer ="Yes: " + str(proba_yes) + ", No: " + str(proba_no)

#    return json.dumps({"response": answer})

#HELPER FUNCTIONS FOR ABOVE FUNCTIONS
def filter_text(unfiltered_text):
    filtered_text = list([char for char in unfiltered_text if (char in string.punctuation or char.isalpha())])
    return "".join(filtered_text)

def parse_new_knowledge(source_file):
    database_insert("INSERT INTO", "Sources", [("Title", source_file), ("Confidence Rating", 6), ("Location", source_file)])

    for line in open(source_file, 'r', encoding="latin-1").readlines():
        for sentence in nltk.sent_tokenize(line):
            associated_nouns = []
            associated_verbs = []
            tokenized_words = nltk.word_tokenize(sentence)
            tokenized_words_filtered = list([word for word in nltk.pos_tag(tokenized_words) if(word[1] in nouns_pos or word[1] in verbs_pos)])
            word_position = -1
            noun_status = 0
            source_sentence = ""
            for word in tokenized_words_filtered:
                noun_status = int(word[1] in nouns_pos)
                if (database_select("Words", "Word", word[0], 0) == False):
                    database_insert("INSERT INTO", "Words", [("Word", word[0]), ("Synonym List", get_synset(word[0])), \
                    ("Hyponym List", get_hyponyms(word[0])), ("Hypernym List", get_hypernyms(word[0])), ("Noun", noun_status), ("Source", source_file)])

                word_id = database_select("Words", "Word", word[0], 1)
                if(noun_status == 1):
                    associated_nouns.append(word_id)
                else:
                    associated_verbs.append(word_id)

            for i in range(len(tokenized_words)):
                if(i != len(tokenized_words) - 1):
                    source_sentence += tokenized_words[i] + " "
                else:
                    source_sentence += tokenized_words[i]

            database_insert("INSERT INTO", "Statements", [("Nouns", str(associated_nouns)), ("Verbs", str(associated_verbs)), \
            ("Source Sentence", source_sentence), ("Source", str(source_file)), ("NOVA", str(list([word[0] for word in tokenized_words_filtered])))])

def database_insert(command, table, vals):
    db_string = command + " " + table + "( "
    if (len(vals[0][0].split()) > 1):
        db_string += ", [" + vals[0][0] + "]"
    else:
        db_string += vals[0][0]

    for item in vals[1:]:
        if (len(item[0].split()) > 1):
            db_string += ", [" + item[0] + "]"
        else:
            db_string += ", " + item[0]

    db_string += ") VALUES(?"
    for i in range(len(vals[1:])):
        db_string += ", ?"

    db_string += ")"

    tuple_list = []
    for item in vals:
        tuple_list.append(str(item[1]))

    print(db_string)

    cursor.execute(db_string, tuple(tuple_list))
    conn.commit()
    return int(cursor.execute("SELECT @@IDENTITY as id").fetchone()[0])

def database_select(table, field, val, mode):
    if (mode == 0): #Get Existence
        return cursor.execute("SELECT COUNT(1) FROM " + table + " WHERE " + field + " = ?", (val)).fetchall()[0][0] > 0

    elif(mode == 1): #Get Location
        return str(cursor.execute("SELECT ID FROM " + table + " WHERE " + field + " = ?", (val)).fetchall()[0][0])

def get_synset(word):
    synonym_list = []
    if(len(wordnet.synsets(word)) > 0):
        for i in range(len(wordnet.synsets(word))):
            if(wordnet.synsets(word)[i].lemmas()[0].name() not in synonym_list):
                synonym_list.append(wordnet.synsets(word)[i].lemmas()[0].name())

        return(str(synonym_list))

    else:
        return "N/A"

def get_hyponyms(word):
    hyponym_list = []
    if(len(wordnet.synsets(word)) > 0):
        for item in wordnet.synsets(word)[0].hyponyms():
            if(str(item.lemmas()[0].name not in hyponym_list)):
                hyponym_list.append(lem.lemmatize(str(item.lemmas()[0].name()), "v"))
        return(str(hyponym_list))

    else:
        return "N/A"

def get_hypernyms(word):
    hypernym_list = []
    if(len(wordnet.synsets(word)) > 0):
        for item in wordnet.synsets(word)[0].hypernyms():
            if(str(item.lemmas()[0].name not in hypernym_list)):
                hypernym_list.append(lem.lemmatize(str(item.lemmas()[0].name()), "v"))
        return(str(hypernym_list))

    else:
        return "N/A"

#def get_relevant_context(question):


app.run(debug=True)
