from flask import Flask, render_template, request
import os
import base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.oauth2 import service_account
from google.cloud import firestore

# Membuat instance Flask-Core
app = Flask(__name__)

# Konfigurasi folder unggahan menggunakan direktori sementara
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Konfigurasi Vertex AI
PROJECT = "97657008905"
ENDPOINT_ID = "7602895307164090368"
LOCATION = "us-central1"
CREDENTIALS_PATH = "testing-jinam-446907-fa4c2327a45b.json"

# Konfigurasi Firebase
FIREBASE_PROJECT = "jinam-446907"
FIREBASE_CREDENTIALS_PATH = "jinam-446907-firebase-adminsdk-h5ozk-83579d2459.json"

# Fungsi prediksi
def predict_image_classification_sample(filename):
    # Menyeting kredensial secara eksplisit
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options, credentials=credentials
    )

    # Membaca file gambar
    with open(filename, "rb") as f:
        file_content = f.read()

    # Encoding gambar ke Base64
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5, max_predictions=5
    ).to_value()

    # Prediksi
    endpoint = client.endpoint_path(
        project=PROJECT, location=LOCATION, endpoint=ENDPOINT_ID
    )
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)

    predictions = []
    for prediction in response.predictions:
        for label in prediction["displayNames"]:
            predictions.append({
                "label": label,
                "confidence": prediction["confidences"][prediction["displayNames"].index(label)]
            })

    # Simpan hasil prediksi ke Firestore
    firebase_credentials = service_account.Credentials.from_service_account_file(FIREBASE_CREDENTIALS_PATH)
    firestore_client = firestore.Client(project=FIREBASE_PROJECT, credentials=firebase_credentials)
    doc_ref = firestore_client.collection("predict").document()
    doc_ref.set({"predictions": predictions})

    return predictions

# Route halaman utama
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Ambil file gambar yang diunggah
        file = request.files.get("file")
        if not file:
            return "File gambar tidak diunggah!", 400

        try:
            # Simpan file gambar sementara
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            # Lakukan prediksi
            predictions = predict_image_classification_sample(filename=image_path)

            # Hapus file setelah digunakan
            os.remove(image_path)

            # Tampilkan hasil prediksi
            return render_template("results.html", predictions=predictions)
        except Exception as e:
            import traceback
            traceback.print_exc()  # Cetak log kesalahan untuk debugging
            return str(e), 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
