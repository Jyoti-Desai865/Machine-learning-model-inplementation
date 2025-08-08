# Machine-learning-model-inplementation

*COMPANY*: CODETECH IT SOLUTION

*NAME*: JYOTI PRADEEP DESAI

*INTERN ID*: CT04DH2737

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

It is implementing a Machine Learning model to classify SMS messages as either spam or ham (not spam). This is a common text classification problem, and it demonstrates the use of supervised learning with natural language processing (NLP) techniques. The goal is to train a model that can analyze the text of a message and accurately predict whether it's spam.
Steps Performed:
Importing Libraries:
We started by importing essential Python libraries such as pandas, numpy, matplotlib, and seaborn for data handling and visualization.
From scikit-learn, we imported tools for model training (train_test_split), text vectorization (CountVectorizer), and model evaluation (accuracy_score, confusion_matrix, classification_report).
The classifier used is MultinomialNB from sklearn.naive_bayes.
Dataset Loading:
The dataset used in this task is a TSV file containing SMS messages labeled as either spam or ham. It was loaded directly from a GitHub URL using pandas.read_csv() with proper parameters for tab-separated values.
Label Encoding:
The labels 'ham' and 'spam' were mapped to numerical values 0 and 1, respectively. This encoding is essential for training the machine learning model.
Splitting the Data
The dataset was split into training and testing sets using an 80-20 ratio to ensure proper model validation.
Text Vectorization:
The messages were converted into numerical features using the Bag-of-Words model via CountVectorizer. This converts raw text into a matrix of token counts, which is a necessary step for applying machine learning algorithms.
Model Training:
A MultinomialNB (Naive Bayes) model was trained on the training data. This classifier is suitable for text classification problems involving discrete features like word counts.
Model Evaluation:
The trained model was evaluated using accuracy, confusion matrix, and a classification report to assess how well it performed on the test data. The confusion matrix provided a clear view of how many messages were correctly and incorrectly classified.
Tools Used:
Python 3.x for programming.
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn.
Jupyter Notebook or VS Code for writing and running the notebook.
pip for installing required packages.
Visual Studio Code (VS Code) was used as the editor platform. The Jupyter extension was added to support .ipynb (Jupyter notebook) files.
This task is highly applicable in real-world spam detection systems used by messaging apps and email services. It introduces basic but powerful concepts in machine learning, especially in the domain of Natural Language Processing. It’s useful for beginner to intermediate-level ML projects and can be extended to more complex applications like phishing detection, review classification, and more.

*OUTPUT*:

<img width="820" height="327" alt="Image" src="https://github.com/user-attachments/assets/b540a23e-df54-49f5-9c72-c599167187de" />

<img width="539" height="455" alt="Image" src="https://github.com/user-attachments/assets/6c80f214-8722-42eb-8dd2-18b96243075c" />
