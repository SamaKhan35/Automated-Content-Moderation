# Automated Content Moderation System

WebApp Link: [Automated Content Moderation Web App](https://moderator-415722.ue.r.appspot.com/) 
(It might take around 30-40 seconds to load due to the initialization of various components, such as loading machine learning models, establishing connections with external APIs, and configuring the backend server.)

## Introduction
This project implements an automated content moderation system that classifies textual content as appropriate or inappropriate for public display. Leveraging machine learning models (LSTM and BERT) and natural language processing techniques, the system analyzes textual data in real-time to provide classification results. Additionally, the project extends its functionality to extract text from images and memes (using Google Cloud Vision API ), enabling classification based on the extracted textual content.

## Files Included
1. `Data_Preprocessing.ipynb`: Jupyter notebook for loading and preprocessing the dataset (`hate_offensive_data.csv`) to create a new CSV file (`preprocessed_data.csv`).
2. `LSTM.ipynb`: Jupyter notebook for training and evaluating the LSTM model for text classification using the preprocessed data.
3. `BERT.ipynb`: Jupyter notebook for training and evaluating the BERT model for text classification using the preprocessed data.
4. `requirements.txt`: File listing the Python libraries required to be installed before executing the code.
5. `Flask_app/main.py`: Python file for the Flask application. This file handles text and image inputs through a web browser, loads the trained BERT model, and accesses the Google Cloud Vision API for text extraction from images.
6. `Flask_app/templates/index5.html`: HTML page for the Flask application.
7. `Flask_app/app.yaml`: Configuration file required for deploying the Flask application on Google App Engine.

## Steps to Simulate the Project
1. Download the repository or clone it using the following command: git clone https://github.com/NavjotDS/Moderator.git
2. Install the required Python libraries listed in `requirements.txt`: `pip install -r requirements.txt`
3. Run `Data_Preprocessing.ipynb` to preprocess the dataset and generate the `preprocessed_data.csv` file.
4. Open and run `LSTM.ipynb` to train and evaluate the LSTM model for text classification using the preprocessed data.
5. Open and run `BERT.ipynb` to train and evaluate the BERT model for text classification using the preprocessed data.
6. Navigate to the `Flask_app` directory and run the Flask application by executing the following command: `python main.py`
7. Access the Flask application through a web browser by visiting `http://localhost:5000/`. You can input text or upload images to receive classification results based on the content provided.

Note: 
- Trained model (BERT) files are not included in the repository and need to be trained separately using the provided notebooks. 
- BERT training may take time. We had used Google Colab to train it faster. 
- Also, please change the file locations in the python files accordingly.

## Results
To assess the performance of our classification models, we conducted a comprehensive evaluation using key metrics such as accuracy, precision, recall, and the F1 score. These metrics provide valuable insights into the abilities of classifiers to accurately categorize textual content as either inappropriate or appropriate for public display.

| Metric    | LSTM       | BERT       |
|-----------|------------|------------|
| Accuracy  | 93.65%     | 95.62%     |
| Precision | 95.97%     | 97.20%     |
| Recall    | 96.41%     | 97.55%     |
| F1 Score  | 96.19%     | 97.37%     |

Through these evaluations, BERT demonstrated superior performance over LSTM, particularly in terms of accuracy, precision, recall, and F1 score. BERT's enhanced capabilities make it the preferred choice for accurate and reliable content moderation tasks.

## Deployment
The web application developed as part of this project has been deployed using Google App Engine, a cloud-based hosting platform known for its scalability and reliability. This deployment ensures seamless access and global availability of the content moderation system, meeting the demands of varying user loads and usage patterns.
WebApp Link: [Automated Content Moderation Web App](https://moderator-415722.ue.r.appspot.com/) 

## Contributors

- [Navjot Singh](https://github.com/NavjotDS)
- [Samaun Sarwar Khan](https://github.com/SamaKhan35)
- [Umang Sood ](https://github.com/Umang-us98)
# Automated-Content-Moderation
# Content-Moderation
