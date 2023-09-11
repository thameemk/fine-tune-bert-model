# Fine-Tuning BERT Model to Identify Actions in Search Queries

### 1. Data Collection
    - Create a custom data / download a dataset that contains unstructured text data.
    
    - Annotate the data with custom labels that represent structred information
    
    Eg:- 
        dataset = [
            {
                "text": "I want to search for information on BERT models.",
                "labels": ["search"]
            },
            {
                "text": "Please send me the latest report.",
                "labels": ["send"]
            },
            {
                "text": "Download the user manual from the website.",
                "label": ["download"]
            },
            {
                "text": "I want to search for information on BERT models and send the documentation to email",
                "label": ["search","send"]
            },
            {
                "text": "lookup for information on BERT models and download the documentation as a pdf",
                "label": ["search","download"]
            },
        ]
        
### 2. Preprocessing the data
    - Lowercasing and removing of irrevelent information
    - Converting special character into meaningful sentences
    - Tokenization of data
### 3. Data Splitting
    - Split the dataset into training (70%), validation (15%) and testing (15%) sets.
### 4.BERT Tokenization
    - Use a BERT tokenizer to convert your text data into BERT tokenized format.
### 5.Model Selection
    - Choose a suitable BERT model 
### 6.Fine-Tuning Of BERT Model 
    - 
### 7.Evaluation
    - Evaluate the fine-tuned model on the test set.
### 8.Developing APIs & Deployment
    - Save the tuned BERT model for inference. 
    - Develope APIs to get user input and use the saved model for generating the results.
    - Develop the functionilities like send email, send sms, download as pdf etc. 
    - Deploy tha application and AI engine for BERT fine tuning.

### 9.Maintenance
    - Monitor the application for quickly resolving bugs.
    - Train the model with new datasets for improvising the results.


