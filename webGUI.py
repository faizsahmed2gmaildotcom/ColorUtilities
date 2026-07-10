from flask import Flask, request, render_template, send_from_directory, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from fabricDetector import predictImage
from config import config
from archived import spreadsheetLib as sL
import atexit

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

fancy_colors = sorted(config["fancy-colors"].keys())
primary_colors = sorted(config["primary-colors"].keys())
patterns = sorted(config["patterns"]["shirting"])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    images = []
    for file in sorted(files, key=lambda f: f.filename):
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_name = os.path.splitext(filename)[0].removesuffix(config["general"]["main_image_suffix"])
            images.append({
                'filename': filename,
                'img_name': img_name,
                'src': f'/static/uploads/{filename}',
                'fancy': 'loading...',
                'primary': 'loading...',
                'secondary1': 'loading...',
                'secondary2': 'loading...',
                'pattern': 'loading...',
                'weave': 'loading...'
            })
    return render_template('results.html', images=images, fancy_colors=fancy_colors, primary_colors=primary_colors,
                           patterns=patterns)


@app.route('/process_image/<filename>')
def process_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    fancy, primary, secondary_list, pattern = predictImage(filepath)
    secondary1 = secondary_list[0] if len(secondary_list) > 0 else ''
    secondary2 = secondary_list[1] if len(secondary_list) > 1 else ''
    return jsonify({
        'fancy': fancy,
        'primary': primary,
        'secondary1': secondary1,
        'secondary2': secondary2,
        'pattern': pattern
    })


@app.route('/save', methods=['POST'])
def save():
    for key, value in request.form.items():
        if key.startswith('img_name_'):
            idx = key.split('_')[-1]
            img_name = value
            fancy = request.form.get(f'fancy_{idx}')
            primary = request.form.get(f'primary_{idx}')
            secondary1 = request.form.get(f'secondary1_{idx}')
            secondary2 = request.form.get(f'secondary2_{idx}')
            pattern = request.form.get(f'pattern_{idx}')
            secondary = ', '.join([s for s in [secondary1, secondary2] if s])
            new_row = sL.Row()
            new_row.update(sku=img_name, Product_Name=fancy, color_filter_primary=primary,
                           color_filter_secondary=secondary, pattern=pattern)
            sL.insertRow(new_row)
    sL.save()

    return send_file("result.xlsx", as_attachment=True, download_name='results.xlsx',
                     mimetype='application/vnd.openpyxlformats-officedocument.spreadsheetml.sheet')


@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory('static', filename)


def cleanup():
    from shutil import rmtree
    print(f"Clearing {UPLOAD_FOLDER}...")
    rmtree(UPLOAD_FOLDER)
    os.mkdir(UPLOAD_FOLDER)
    print(f"Cleared {UPLOAD_FOLDER}!")


atexit.register(cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
