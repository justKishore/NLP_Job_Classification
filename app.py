from flask import Flask, render_template,request, redirect,url_for,session
import difflib
import numpy as np
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
import joblib

import os
import json

import json

app = Flask(__name__)


app.secret_key = os.urandom(20) # secret key


def read_data(): # function to json data, Note: Rather than html, json file is used as it is more dynamic and in future it can easily replaced with database object (e.g. MongoDB object), so data will be stored in DB
    try:
        with open('templates/data.json', 'r') as f:
            job_data = json.load(f)
        return job_data
    except:
        return []


# function to update json file, so new job data will be added
def post_job_to_json(data):
    # Read existing data
    job_data = read_data()
    # Assign a unique ID so it will route based on id rahter than title, as two jobs can have same title
    data['id'] = str(len(job_data) + 1)

    # Append the new data to the existing data
    job_data.append(data)

    # Write the updated data back to the JSON file
    with open('templates/data.JSON', 'w') as f:
        json.dump(job_data, f, indent=2)
    return data['id']




# docvecs function is taken from week 11 materials
def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs

def golve_load(path_to_glove_file):
    glove_wv = {} # initialise an empty dcitionary
    with open(path_to_glove_file) as f: # open the txt file containing the word embedding vectors
        for line in f:
            word, coefs = line.split(maxsplit=1) # The maxsplit defines the maximum number of splits. 
                                                # in the above example, it will give:
                                                # ['population','0.035182 1.4248 0.9758 0.1313 -0.66877 0.8539 -0.11525 ......']
            coefs = np.fromstring(coefs, "f", sep=" ") # construct an numpy array from the string 'coefs', 
                                                    # e.g., '0.035182 1.4248 0.9758 0.1313 -0.66877 0.8539 -0.11525 ......'
            glove_wv[word] = coefs # create the word and embedding vector mapping

    print("Found %s word vectors." % len(glove_wv))
    return glove_wv



# Routes Start form here 

# home route
@app.route('/') 
def index(): 
    job_data = read_data()
    return render_template('index.html',jobs=job_data,route=request.endpoint)


# search route
# here user can search based on keyword, job category (classification) and location. Idea from seek.com.au
@app.route('/search', methods=['POST'])
def search():
    job_data = read_data()
    what = request.form.get('what')
    category = request.form.get('category')
    where = request.form.get('where')

    if( what=='' and where=='' and category=='Any'):
        return redirect(url_for('any_classification'))
    
    else:
        # Initialize with all data
        filtered_data = job_data
        close_matches = ''
        closest_matches = []  # Initialize an empty list for closest matches
        similar_jobs = []
        
        # Filter data based on user search inputs

        # Note: Users may or may not use any inputs
        if what:
            filtered_data = [job for job in filtered_data if what.lower() in job['title'].lower() or what.lower() in job['description'].lower()]
            
            # Note : Not part of requiremnt, just showing similar results as improvisation. Skip if required
            for job in job_data:
                sim = difflib.SequenceMatcher(None, what.lower(), job['title'].lower()).ratio()
                if sim >= 0.6:
                    similar_jobs.append(job)

        if category != 'Any':
            filtered_data = [job for job in filtered_data if job['category'] == category]

        if where:
            filtered_data = [job for job in filtered_data if where.lower() in job['location'].lower()]

        
        if len(similar_jobs)==0:
            similar_jobs = None # if no similar jobs found, set it to None, so it is easy to handle in font end
        else:
            similar_jobs = [job for job in similar_jobs if job not in filtered_data] # Need not show searched results inside similar jobs

        return render_template('index.html', jobs=filtered_data,length=len(filtered_data),route=request.endpoint, similar_jobs = similar_jobs)


# just an extra route if user click search without any input fields
@app.route("/any-classification")
def any_classification(): 
        
        job_data = read_data()
        length = len(job_data)
        return render_template('index.html', jobs=job_data, length=length, route=request.endpoint)

# if user just clears <doc> from /jobs/<doc> it will go to home rather than throwing error
@app.route("/jobs")
def jobs():
    return redirect(url_for('index'))


# Route to show expanded details of each job data

# here job data is handeled by creating and custom unique ID. Assign a unique ID so it will route based on id rahter than title, as two jobs can have same title

@app.route("/jobs/<doc>")
def show_job(doc):

    job_data = read_data()

    for job in job_data:
        if (job['id'] == doc):
            return render_template('job.html',job = job)
    # return(doc)
    return redirect(url_for('index'))



# Route for employer or to use Create New Job Listing functionality
@app.route("/employer")
def employee():
    if 'username' in session:
        return render_template('employer-step1.html')
    else:
        return redirect('/login')


@app.route('/create_job', methods=['POST'])

# Given that this route is a POST route accessible exclusively through "employer-step1.html,"  which is only accessible when user is logged in (check "/employer")
# So, there's no need to perform an additional login check.

def create_job():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        company = request.form['company']
        location = request.form['location']
        salary = request.form['salary']
        job_type = request.form['job_type']

        # prediction based on description

        # try catch helps in avoiding expect or unexpected error during perdict operations. If error's rise, it is shown as alert rather than breaking the page
        try:
            tokenized_data = description.split(' ') # tokenizing user typed descriprion

            glove_model = KeyedVectors.load("./NLP/generated_model/glove.model") # golve model, here we already have as wv


            # desc_FT_wv= desc_FT.wv

            glove_dvs = docvecs(glove_model, [tokenized_data])
            lr_model = "./NLP/generated_model/logistic_regression_model.pkl"

            model = joblib.load(lr_model)
            y_pred = model.predict(glove_dvs) # Predict the label of tokenized_data
            print(y_pred)
            y_pred = y_pred[0]
            recommended_category = y_pred

            # the some commented files are form my submitted version, the new glove based one is done during sem holidays for the glove model I generated




        except Exception as e:
            print(e)
            recommended_category = -1
        return render_template('employer-step2.html', 
                   title=title,
                   description=description,
                   company=company,
                   location=location,
                   salary=salary,
                   job_type=job_type,
                   recommended_category=recommended_category)

# App to post the job into json file

@app.route("/post_job",methods=['POST'])
def post_job():
    form_data = {}
    for form_key, form_value in request.form.items():
        form_data[form_key] = form_value
    print(form_data)
    id = post_job_to_json(form_data)
    return redirect('jobs/' + id)

# code indpired form week 11 lab
@app.route('/login', methods=['GET', 'POST']) # for employee login
def login():
    if 'username' in session:
        return redirect('/employer')
    else:
        if request.method == 'POST':
            if (request.form['username'] == 'admin') and (request.form['password'] == 'secret'):
                session['username'] = request.form['username']
                return redirect('/employer')
            else:
                return render_template('login.html', error="True")
        else:
            return render_template('login.html')
        

# Route to log out
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect("/")

# Error hadling section

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error_404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error_500.html'), 500