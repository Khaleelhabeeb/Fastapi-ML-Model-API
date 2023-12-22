from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from tensorflow import keras
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse

app = FastAPI()

# Load the trained model
model = keras.models.load_model('./models/model.weights.best.hdf5')

# Define the leaf classes
leaf_class = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# Define the prediction threshold
prediction_threshold = 0.75  # Adjust this value as needed

async def predict_leaf_class(image: UploadFile):
    # Read the image and preprocess it
    img = Image.open(image.file)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)

    # Check if the highest probability is above the threshold
    highest_probability = np.max(prediction)

    if highest_probability < prediction_threshold:
        predicted_class = 'No disease found'
    else:
        predicted_class = leaf_class[np.argmax(prediction)]

    return predicted_class

@app.post("/predict")
async def predict_leaf_disease(image: UploadFile = File(...)):
    predicted_class = await predict_leaf_class(image)
    return JSONResponse(content={"predicted_class": predicted_class})


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with the actual frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)