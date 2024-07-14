from flask import Flask, request, jsonify
import pickle
import numpy as np

with open('Wine_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    features = np.array(data['Alcohol','Malic_Acid','Ash','Ash_Alcanity	Magnesium',	'Total_Phenols','Flavanoids','Nonflavanoid_Phenols','Proanthocyanins',	'Color_Intensity',	'Hue',	'OD280','Proline','Customer_Segment'])
    
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    cluster = kmeans.predict(features)
    
    return jsonify({'cluster': int(cluster[0])})

if __name__ == '__main__':
    app.run(debug=True)