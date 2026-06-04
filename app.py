from flask import Flask, render_template, request, redirect
import pickle
import traceback

app = Flask(__name__, template_folder='templates')

# Load models
try:
    with open("models/all_models.pkl", "rb") as f:
        model_data = pickle.load(f)

    vectorizer = model_data["vectorizer"]
    models = model_data["models"]
    metrics = model_data["metrics"]
except Exception as e:
    print("❌ Model load failed:", str(e))
    traceback.print_exc()
    vectorizer = None
    models = {}
    metrics = {}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        news = request.form.get("news", "")
        if not news.strip():
            return redirect('/')

        vector = vectorizer.transform([news])
        predictions = {}
        accuracies = {}

        for model_name, model in models.items():
            try:
                if model_name == "Linear Regression":
                    pred = model.predict(vector)[0]
                    result = "Real News ✅" if pred > 0.5 else "Fake News ❌"
                else:
                    pred = model.predict(vector)[0]
                    result = "Real News ✅" if pred == 1 else "Fake News ❌"

                predictions[model_name] = result
                accuracies[model_name] = metrics[model_name]['accuracy']
            except Exception as err:
                print(f"Error in model {model_name}: {err}")
                predictions[model_name] = "Error"
                accuracies[model_name] = 0.0

        best_model = max(metrics.items(), key=lambda x: x[1]['f1'])[0]
        final_prediction = predictions[best_model]

        return render_template("result.html",
                               input_text=news,
                               predictions=predictions,
                               accuracies=accuracies,
                               best_model=best_model,
                               final_prediction=final_prediction)

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return "500 - Internal Server Error"

if __name__ == "__main__":
    app.run(debug=True)
