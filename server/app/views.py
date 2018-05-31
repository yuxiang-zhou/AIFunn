import os, traceback
from flask import render_template
from app import app, fitting
from flask import Flask, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path


@app.route('/')
@app.route('/index')
def index():

    return redirect(url_for('upload_file'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            saving_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(saving_path)


            try:
                fitting.fit(saving_path)

                mesh_texture = Path(saving_path)
                filename = mesh_texture.stem + '.render.jpg'

                return redirect(url_for('download',
                                        filename=str(mesh_texture.stem)))
            except Exception as e:
                print('Error: %s'%e)
                traceback.print_exc()

                return '''
                <!doctype html>
                <title>Upload new File</title>
                <p>Failed to Detect Faces</p>
                <h1>Upload new File</h1>
                <form method=post enctype=multipart/form-data>
                  <p><input type=file name=file>
                     <input type=submit value=Upload>
                </form>
                '''
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/download/<filename>')
def download(filename):

    return render_template("download.html",
                           filename=filename)


@app.route('/files/<filename>')
def files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
