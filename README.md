**Synthetic Data Studio Guide: Lending data generation using Free Form API**

**Overview**  

The goal is to generate high-quality synthetic lending data that mirrors real-world patterns while ensuring privacy and utility for training machine learning models. This guide uses Synthetic Data Studio (SDS) Free Form API, assumes SDS is installed, and runs in a session.   
The guide involves five sections: providing examples, synthesizing data, evaluating quality, filtering invalid entries and converting to tabular data, and saving data. 

**Section 1: Example Data**  

*Description:*  

The example data (e.g., \`ExamplesLoanData.json\`) serves as a template for synthetic data generation. It includes sample records with fields like loan amount, interest rate, employment details, and addresses. These examples guide the LLM in understanding valid formats, value ranges, and relationships between variables (e.g., how interest rates correlate with credit history).  

*Key Insights*:  

* **Diversity** is critical: Include edge cases (e.g., low/high income, short/long employment history).    
* **Consistency** in categories: Ensure fields like \`home\_ownership\` use predefined values (RENT, OWN, MORTGAGE, OTHER).    
* **Plausible values**: Avoid outliers (e.g., a \`dti\` ratio of 1000% is unrealistic).    
* **Address formatting**: Ensure addresses follow a consistent structure (e.g., street, city, state, ZIP code).  

*Code:*

Here is an example file for our lending data application.The file needs to be saved as ExamplesLoanData.json

```

[
  {
    "loan_amnt": 10000.00,
    "term": "36 months",
    "int_rate": 11.44,
    "installment": 329.48,
    "grade": "B",
    "subgrade": "B4",
    "emp_title": "Marketing",
    "emp_length": "10+ years",
    "home_ownership": "RENT",
    "annual_inc": 117000.00,
    "verification_status": "Not Verified",
    "issue_d": "Jan-2015",
    "loan_status": "Fully Paid",
    "purpose": "vacation",
    "title": "Vacation",
    "dti": 26.24,
    "earliest_cr_line": "Jun-1990",
    "open_acc": 16.00,
    "pub_rec": 0.00,
    "revol_bal": 36369.00,
    "revol_util": 41.80,
    "total_acc": 25.00,
    "initial_list_status": "w",
    "application_type": "INDIVIDUAL",
    "mort_acc": 0.00,
    "pub_rec_bankruptcies": 0.00,
    "address": "0174 Michelle Gateway\r\nMendozaberg, OK 22690"
  },
  {
    "loan_amnt": 8000.00,
    "term": "36 months",
    "int_rate": 11.99,
    "installment": 265.68,
    "grade": "B",
    "subgrade": "B5",
    "emp_title": "Credit analyst",
    "emp_length": "4 years",
    "home_ownership": "MORTGAGE",
    "annual_inc": 65000.00,
    "verification_status": "Not Verified",
    "issue_d": "Jan-2015",
    "loan_status": "Fully Paid",
    "purpose": "debt_consolidation",
    "title": "Debt consolidation",
    "dti": 22.05,
    "earliest_cr_line": "Jul-2004",
    "open_acc": 17.00,
    "pub_rec": 0.00,
    "revol_bal": 20131.00,
    "revol_util": 53.30,
    "total_acc": 27.00,
    "initial_list_status": "f",
    "application_type": "INDIVIDUAL",
    "mort_acc": 3.00,
    "pub_rec_bankruptcies": 0.00,
    "address": "1076 Carney Fort Apt. 347\r\nLoganmouth, SD 05113"
  },
  {
    "loan_amnt": 15600.00,
    "term": "36 months",
    "int_rate": 10.49,
    "installment": 506.97,
    "grade": "B",
    "subgrade": "B3",
    "emp_title": "Statistician",
    "emp_length": "< 1 year",
    "home_ownership": "RENT",
    "annual_inc": 43057.00,
    "verification_status": "Source Verified",
    "issue_d": "Feb-2015",
    "loan_status": "Fully Paid",
    "purpose": "credit_card",
    "title": "Credit card refinancing",
    "dti": 12.79,
    "earliest_cr_line": "Aug-2007",
    "open_acc": 13.00,
    "pub_rec": 0.00,
    "revol_bal": 11987.00,
    "revol_util": 92.20,
    "total_acc": 26.00,
    "initial_list_status": "f",
    "application_type": "INDIVIDUAL",
    "mort_acc": 0.00,
    "pub_rec_bankruptcies": 0.00,
    "address": "87025 Mark Dale Apt. 269\r\nNew Sabrina, WV 05113"
  }
]

```

**Section 2: Data Generation Via SDS freeform API**

*Description:*  

The code uses Synthetic Data Studio’s freeform API to generate synthetic data based on the examples and a custom prompt. The LLM (e.g., Claude-3-5) creates new records by inferring patterns from the examples and adhering to field descriptions. Here, we can set different parameters such as temperature, Max tokens, Top K, and different seed instructions to generate diverse lending data samples.

*Key Insights:*  

* Clear prompt structure:   
  * Define each field’s purpose and constraints explicitly (e.g., "term must be 36 or 60 months").    
  * Describe fields very precisely and in the expected order matching the examples  
* Parameter tuning:    
  * Temperature: Lower values (e.g., 0.5) reduce randomness for more consistent data; higher values (e.g., 1.0) increase diversity.    
  * Max tokens: Ensure sufficient length to capture all fields.    
  * Top K: Use high values (e.g.) to ensure decoding diversity  
* Reference examples: The API uses \`example\_path\` to mimic real-world distributions. Poor examples lead to flawed outputs.    
* Grounding knowledge: If the LLM does not possess information about some specific fields, for example, the base interest rates during a period, consider adding these facts as part of the prompt.  
* Check prompt syntax: Use a syntax and grammar checker to spot any prompt typos.


  
*Code:*

Here is the code to generate synthetic lending data. Ensure your url (`url = 'https://synthetic-data-generator-oaqw25.ai-workbench.eng-ml-l.vnu8-sqze.cloudera.site/synthesis/freeform'`) is replaced with your project url:

```py
%%time
import requests
import os

# Get API key from environment variable if within CDSW app/session
api_key = os.environ.get('CDSW_APIV2_KEY')


# URL for synthesis
url = 'https://synthetic-data-generator-oaqw25.ai-workbench.eng-ml-l.vnu8-sqze.cloudera.site/synthesis/freeform'


# Add the API key to headers with proper Authorization format
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

# If API key exists, add it to the headers
if api_key:
    headers['Authorization'] = f'Bearer {api_key}'
else:
    print("Warning: No API key provided")

# Payload for data synthesis
payload = {
  "inference_type": "aws_bedrock",
  "is_demo": True,
  "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
  "num_questions": 10,
  "custom_prompt": """
 You need to create profile data for the LendingClub company which specialises in lending various types of loans to urban customers.
 

You need to generate the data in the same order for the following  fields (description of each field is followed after the colon):

loan_amnt: The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
term: The number of payments on the loan. Values are in months and can be either 36 or 60.
int_rate: Interest Rate on the loan
installment: The monthly payment owed by the borrower if the loan originates.
grade: LC assigned loan grade
sub_grade: LC assigned loan subgrade
emp_title: The job title supplied by the Borrower when applying for the loan.*
emp_length: Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
home_ownership: The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
annual_inc: The self-reported annual income provided by the borrower during registration.
verification_status: Indicates if income was verified by LC, not verified, or if the income source was verified
issue_d: The month which the loan was funded
loan_status: Current status of the loan
purpose: A category provided by the borrower for the loan request.
title: The loan title provided by the borrower
dti: A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
earliest_cr_line: The month the borrower's earliest reported credit line was opened
open_acc: The number of open credit lines in the borrower's credit file.
pub_rec: Number of derogatory public records
revol_bal: Total credit revolving balance
revol_util: Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
total_acc: The total number of credit lines currently in the borrower's credit file
initial_list_status: The initial listing status of the loan. Possible values are – W, F
application_type: Indicates whether the loan is an individual application or a joint application with two co-borrowers
mort_acc: Number of mortgage accounts.
pub_rec_bankruptcies: Number of public record bankruptcies
address: The physical address of the person


""",
  "model_params": {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 250,
    "max_tokens": 8192
  },
  "use_case": "custom",
  "topics": [
    "Financial data"
  ],
  "example_path": "ExamplesLoanData.json"
}

# Make the POST request
response = requests.post(url, headers=headers, json=payload)

# Display the response
print(response.status_code)
print(response.json())

```

### **Section 3: LLM-As-A-Judge Evaluation of Lending Synthetic Data**

*Description:*

The evaluation step uses another LLM (e.g., Claude-3-7) to score generated rows (1–5) based on realism and internal consistency. The LLM checks dependencies (e.g., interest rates matching the `issue_d` period and borrower risk profile), whether the outputs make sense, and gives a final score and justification which can be used to filter bad data.

*Key Insights:*

* Prompt information: Give the LLM all the information it needs to score the samples, including the definitions of the fields  
* Prompt checks: Do give the LLM examples on what to check. For example, asking the LLM specifically to check for data co-dependencies, temporal consistency and field validity improves evaluation quality.  
* Parameter tuning: Use a low temperature value to target accuracy rather than diversity. This allows the LLM to score the samples with its most accurate answers.  
* Scoring criteria: Specify explicitly how the LLM will score and justify the examples.  
* Scoring range: State the scoring range (values 1-5) to ensure consistency of the scoring output.

*Code:*

The file `"import_path": "freeform_data_claude_20250410T005129890_test.json”` needs to be replaced with the output of the previous step. Here is a code sample for evaluating the data:

```py
import requests
import os
#********************Accessing Application**************************
# Get API key from environment variable if withinin CDSW app/session.
# To get your API key for using outside CDSW app/session follow given link.
# https://docs.cloudera.com/machine-learning/cloud/api/topics/ml-api-v2.html
api_key = os.environ.get('CDSW_APIV2_KEY')


# Below is your application API URL, you can look at swagger documentation for all existing # endpoints for current application
# https://<application-subdomain>.<workbench-domain>/docs--> will take user to swagger documentaion
# Link to application can be found on application details page within CAI Workbench.


# URL for evaluation
url = 'https://synthetic-data-generator-oaqw25.ai-workbench.eng-ml-l.vnu8-sqze.cloudera.site/synthesis/evaluate_freeform'

# Add the API key to headers with proper Authorization format
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'  # Format as specified in the documentation
}   

# The prompt for evaluation
custom_prompt = """Below is financial data synthesized by an LLM which have the following fields (description of each field is followed after the colon):

loan_amnt: The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
term: The number of payments on the loan. Values are in months and can be either 36 or 60.
int_rate: Interest Rate on the loan
installment: The monthly payment owed by the borrower if the loan originates.
grade: LC assigned loan grade
sub_grade: LC assigned loan subgrade
emp_title: The job title supplied by the Borrower when applying for the loan.*
emp_length: Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
home_ownership: The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
annual_inc: The self-reported annual income provided by the borrower during registration.
verification_status: Indicates if income was verified by LC, not verified, or if the income source was verified
issue_d: The month which the loan was funded
loan_status: Current status of the loan
purpose: A category provided by the borrower for the loan request.
title: The loan title provided by the borrower
dti: A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
earliest_cr_line: The month the borrower's earliest reported credit line was opened
open_acc: The number of open credit lines in the borrower's credit file.
pub_rec: Number of derogatory public records
revol_bal: Total credit revolving balance
revol_util: Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
total_acc: The total number of credit lines currently in the borrower's credit file
initial_list_status: The initial listing status of the loan. Possible values are – W, F
application_type: Indicates whether the loan is an individual application or a joint application with two co-borrowers
mort_acc: Number of mortgage accounts.
pub_rec_bankruptcies: Number of public record bankruptcies
address: The physical address of the person:

Rate the quality of the generated whether they reflect real financial data and the variables make sense when considered to together.

For example, int_rate which represents interest rate should reflect real world interest rates during the issue_d period (The month which the loan was funded) but also reflect the risk profile of the person, dti, earliest_cr_line and other fields.
For example, the installment need to make sense when considering the loan amount, the term and the int_rate.

Make sure you analyze as many condependence between the fields and subtract 1 point for each co-dependency that does not reflect real data.
Subtract 2 points if there is a mistake in the field keys or field values.
Subtract 2 points for each other mistake you see with the data.
Give a score rating 1-5 for the given data. 
"""


examples = [
]


# Model parameters
model_params = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 250,
    "max_tokens": 4096
}

payload = {
    "export_type": "local",
    "display_name": "LendingData",
    "import_path": "freeform_data_claude_20250410T005129890_test.json",
    "import_type": "local",
    "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "use_case": "custom",
    "is_demo": True,
    "custom_prompt": custom_prompt,
#    "examples": examples,
    "model_params": model_params
}

response = requests.post(url, headers=headers, json=payload)

# Print the response
print(response.status_code)
print(response.json())

```

### **Section 4: Filtering and Conversion to Tabular Data**

*Description:*

After evaluation, only high-scoring rows (e.g., \>3.5) are retained. The data is converted into a Pandas DataFrame for analysis and preprocessing.

*Key Insights:*

* **Threshold optimization**: Analyze the score distribution to avoid excluding valid data or retaining poor-quality entries. Select a threshold to balance data quality  and data quantity.  
* **Field retention**: Ensure critical fields (e.g., `loan_amnt`, `int_rate`) are present and formatted correctly (e.g., numeric vs. string).

*Code:*

Here is the code to filter bad data using LLM-as-a-judge and converting it to tabular format (In this example, we use a threshold 3.5 to balance data quality and quantity.):

```py
AllRows=[]
for i in response.json()['result']['evaluated_rows']:
  del i['row']['Seeds']
  if i['evaluation']['score'] > 3.5:
      AllRows.append(i['row'])

import pandas as pd
df=pd.DataFrame.from_records(AllRows)
df
```

### **Section 5: Saving to CSV**

*Description:*

The final cleaned data is saved as a CSV file for downstream use (e.g., model training).

*Key Insights:*

- **Format consistency**: Use a delimiter (e.g., tab-separated `\t`) that avoids conflicts with data fields (e.g., commas in addresses).


*Code:*

Here is the code to save the data to csv format for further processing:

```py
OutputFile='data_tab_separated.csv'
df.to_csv(OutputFile, sep='\t')

```

### **Final Key Takeaways**

1. *Quality starts with examples*: Garbage-in, garbage-out applies here.  
2. *Prompt engineering is critical*: Clear instructions and constraints ensure LLMs generate realistic data.  
3. *Evaluation drives filtering*: Use LLM-as-a-judge to eliminate implausible entries.  
4. *Iterate*: Refine examples, prompts, and thresholds based on evaluation results and downstream model performance.
