# FastAPI Leaf Disease Prediction API

This FastAPI application serves as an API for predicting leaf diseases in images using a pre-trained convolutional neural network (CNN) model. The model is based on TensorFlow's Keras and has been trained to classify images into three classes of rice leaf diseases: Bacterial leaf blight, Brown spot, and Leaf smut.

## How it Works

1. **Model Loading**: The application loads a pre-trained CNN model from the specified path (`./models/model.weights.best.hdf5`).

2. **Leaf Classes**: The leaf disease classes are defined as follows:
   - Bacterial leaf blight
   - Brown spot
   - Leaf smut

3. **Prediction Threshold**: A prediction threshold is set to determine whether a disease is confidently predicted. Adjust the `prediction_threshold` value as needed.

4. **Endpoint**: The API provides a single endpoint `/predict` that accepts image uploads for disease prediction.

5. **Prediction Functionality**: The `predict_leaf_class` function reads the uploaded image, preprocesses it, and makes predictions using the loaded model. If the highest predicted probability is below the threshold, it indicates that no disease is found. Otherwise, the predicted disease class is returned.

6. **CORS Middleware**: Cross-Origin Resource Sharing (CORS) middleware is added to allow cross-origin requests, enabling interaction with the API from different domains.

## How to Use

### Prediction Endpoint

- **Endpoint URL**: `http://your-api-domain/predict`
- **HTTP Method**: POST
- **Request Parameter**:
  - `image`: Upload an image file for leaf disease prediction.

- **Example Usage**:
  ```python
  import requests
  from io import BytesIO
  from PIL import Image

  # Replace 'your-api-domain' with the actual domain or IP address
  api_url = "http://your-api-domain/predict"
  
  # Load an image for prediction
  image_path = "path/to/your/image.jpg"
  image = Image.open(image_path)

  # Prepare image for upload
  img_byte_array = BytesIO()
  image.save(img_byte_array, format="JPEG")
  image_data = {"image": ("image.jpg", img_byte_array.getvalue(), "image/jpeg")}

  # Make a prediction request
  response = requests.post(api_url, files=image_data)

  # Get the predicted class
  result = response.json()
  predicted_class = result["predicted_class"]
  print(f"Predicted Class: {predicted_class}")
