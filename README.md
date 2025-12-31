# customer-churn-behavior-analysis
End-to-end customer churn analysis identifying high-risk customer segments through tenure-based behavioral patterns and revenue signals using Python.


## Business Problem

Customer churn directly impacts revenue, growth, and long-term sustainability.  
While many analyses identify *who* churned, businesses often struggle to understand:

a) When churn happens in the customer lifecycle  
b)  Which combinations of services, contracts, and payment methods increase risk  
c) How pricing influences customer exit decisions  
d) Which churn segments cause the highest revenue loss  

**The goal of this project was to diagnose churn behavior, not merely describe it.**


## Dataset Summary

1. **Total records:** 7,043 customers  
2. **Features:** Demographics, tenure, services, contracts, payment methods, charges  
3. **Target variable:** Churn (Yes / No)  



## Key Engineered Features

I) Customer lifecycle segmentation (New, Mid, Long tenure)  
II) Cleaned and standardized numerical charge variables  
III) Behavioral grouping across services and billing methods  



## Analytical Approach

The analysis followed a structured, business-driven flow:

### 1. Data Cleaning & Preparation
I) Removed non-impactful features  
II) Converted monetary fields to numeric  
III) Handled missing and inconsistent values  

### 2. Customer Lifecycle Segmentation
Created tenure-based segments:
I) New customers  
II) Mid-tenure customers  
III) Long-tenure customers  

### 3. Behavioral Churn Analysis
Analyzed churn patterns across:
I) Contract types  
II) Payment methods  
III) Internet services  
IV) Billing preferences  

### 4. Revenue-Based Insights
I) Monthly charge distribution of churned customers  
II) Total revenue loss across tenure stages  
III) Identification of high-risk pricing bands  

### 5. Insight Validation
I) Cross-analysis of multiple high-risk factors combined  
II) Pattern consistency across customer segments  



## Key Insights

- **New customers are the most vulnerable to churn, especially within the first few months**  
- **Month-to-month contracts show significantly higher churn across all tenure stages**  
- Electronic check payment method is strongly associated with churn behavior  
- Fiber optic service customers churn more frequently than DSL or no-internet users  
- Customers with monthly charges in the **~68–103 range** exhibit the highest churn rate  
- Long-tenure customers churn less frequently but cause higher revenue loss when they do  

These patterns indicate that **churn is behavior-driven, not random**.



## High-Risk Customer Profile (Observed Pattern)

Customers most likely to churn typically share the following combination:

- Month-to-month contract  
- Electronic check payment method  
- Fiber optic internet service  
- Paperless billing enabled  
- Higher monthly charges  
- Early or mid lifecycle stage  

This combination consistently appeared across multiple churn segments.



## Business Recommendations

Based on the analysis, the following actions could reduce churn:

- Strengthen onboarding for new customers within the first few months  
- Encourage longer-term contracts through targeted incentives  
- Promote auto-payment options to reduce friction from electronic check usage  
- Review fiber optic pricing and service experience  
- Flag customers entering high-risk pricing bands for proactive retention outreach  
- Prioritize retention of long-tenure customers due to higher revenue impact  



## Tools & Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  
