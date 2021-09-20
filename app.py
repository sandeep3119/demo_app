from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


params_path = 'params.yaml'
webapp_root = "webapp"


static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")


app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    print(prediction)
    return prediction[0]


def api_response(request):
    try:
        data = np.array([list(request.values())])
        response = predict(data)
        response = {"response": response}
        return response
    except Exception as e:
        logging.error(e)
        error = {'error': "Something Went Wrong"}
        return error


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            data = dict(request.form).values()
            data = [list(map(float, data))]
            response = predict(data)
            return render_template('index.html', response=response)
        except Exception as e:
            logging.error(e)
            error = {'error': "Something Went Wrong"}
            return render_template("404.html", error=error)
    else:
        return render_template('index.html')


@app.route('/api/v1', methods=['POST'])
def getApiPredictions():
    try:
        app.logger.info(request.json.values())
        response = api_response(request.json)
        return jsonify(response)
    except Exception as e:
        print(e)
        error = {'error': "Something Went Wrong"}
        return render_template("404.html", error=error)



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
