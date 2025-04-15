**Synthetic Data Studio Guide: Lending data generation using Free Form API**

**Overview**  

The goal is to generate high-quality synthetic lending data that mirrors real-world patterns while ensuring privacy and utility for training machine learning models. This guide uses Synthetic Data Studio (SDS) Free Form API, assumes SDS is installed, and runs in a session.   
The guide involves five sections: providing examples, synthesizing data, evaluating quality, filtering invalid entries and converting to tabular data, and saving data. 

**Section 1: Inspect Lending club data**

*Description:*  

This step loads the existing Lending Data for inspection.

*Setup*
- The lending data file is specified in the OutputFile variable.


**Section 2a: Example Data**  

*Description:*  

The example data (e.g., \`ExamplesLoanData.json\`) serves as a template for synthetic data generation. It includes sample records with fields like loan amount, interest rate, employment details, and addresses. These examples guide the LLM in understanding valid formats, value ranges, and relationships between variables (e.g., how interest rates correlate with credit history).  

*Key Insights*:  

* **Diversity** is critical: Include edge cases (e.g., low/high income, short/long employment history).    
* **Consistency** in categories: Ensure fields like \`home\_ownership\` use predefined values (RENT, OWN, MORTGAGE, OTHER).    
* **Plausible values**: Avoid outliers (e.g., a \`dti\` ratio of 1000% is unrealistic).    
* **Address formatting**: Ensure addresses follow a consistent structure (e.g., street, city, state, ZIP code).  

*Example:*

An example file is shown in ExamplesLoanData.json


*Setup*
The lending data file is specified at OutputFile.

**Section 2a: Data Generation Via SDS freeform API**

*Description:*  

The step uses Synthetic Data Studio’s freeform API to generate synthetic data based on the examples and a custom prompt. The LLM creates new records by inferring patterns from the examples and adhering to field descriptions. Here, we can set different parameters such as temperature, Max tokens, Top K, and different seed instructions to generate diverse lending data samples.

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


**Section 2b: Example Data**  

*Description:*  

The example data (e.g., \`ExamplesLoanData.json\`) serves as a template for synthetic data generation. It includes sample records with fields like loan amount, interest rate, employment details, and addresses. These examples guide the LLM in understanding valid formats, value ranges, and relationships between variables (e.g., how interest rates correlate with credit history).  

*Key Insights*:  

* **Diversity** is critical: Include edge cases (e.g., low/high income, short/long employment history).    
* **Consistency** in categories: Ensure fields like \`home\_ownership\` use predefined values (RENT, OWN, MORTGAGE, OTHER).    
* **Plausible values**: Avoid outliers (e.g., a \`dti\` ratio of 1000% is unrealistic).    
* **Address formatting**: Ensure addresses follow a consistent structure (e.g., street, city, state, ZIP code).  

*Example:*

An example file is shown in ExamplesLoanData.json


### **Section 3a: LLM-As-A-Judge Evaluation of Lending Synthetic Data**

*Description:*

The evaluation step uses another LLM to score generated rows (1–5) based on realism and internal consistency. The LLM checks dependencies (e.g., interest rates matching the `issue_d` period and borrower risk profile), whether the outputs make sense, and gives a final score and justification which can be used to filter bad data.

*Key Insights:*

* Prompt information: Give the LLM all the information it needs to score the samples, including the definitions of the fields  
* Prompt checks: Do give the LLM examples on what to check. For example, asking the LLM specifically to check for data co-dependencies, temporal consistency and field validity improves evaluation quality.  
* Parameter tuning: Use a low temperature value to target accuracy rather than diversity. This allows the LLM to score the samples with its most accurate answers.  
* Scoring criteria: Specify explicitly how the LLM will score and justify the examples.  
* Scoring range: State the scoring range (values 1-5) to ensure consistency of the scoring output.

*Setup:*

- The file in variable InputFile needs to be replaced with the output file of the generation step. 


### **Section 3b: Filtering and Conversion to Tabular Data**

*Description:*

After evaluation, only high-scoring rows (e.g., \>3.5) are retained. The data is converted into a Pandas DataFrame for analysis and preprocessing.

*Key Insights:*

* **Threshold optimization**: Analyze the score distribution to avoid excluding valid data or retaining poor-quality entries. Select a threshold to balance data quality  and data quantity.  
* **Field retention**: Ensure critical fields (e.g., `loan_amnt`, `int_rate`) are present and formatted correctly (e.g., numeric vs. string).

*Setup:*
- The file in variable LLMJUDGEOUT needs to be replaced with the output file of the LLM-as-a-judge step. 

### **Section 4: Saving to CSV**

*Description:*

The final cleaned data is saved as a CSV file for downstream use (e.g., model training).

*Key Insights:*

- **Format consistency**: Use a delimiter (e.g., tab-separated `\t`) that avoids conflicts with data fields (e.g., commas in addresses).



### **Final Key Takeaways**

1. *Quality starts with examples*: Garbage-in, garbage-out applies here.  
2. *Prompt engineering is critical*: Clear instructions and constraints ensure LLMs generate realistic data.  
3. *Evaluation drives filtering*: Use LLM-as-a-judge to eliminate implausible entries.  
4. *Iterate*: Refine examples, prompts, and thresholds based on evaluation results and downstream model performance.
