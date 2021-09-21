# HOME CREDIT DEFAULT RISK

![Ekran görüntüsü 2021-09-21 231044](https://user-images.githubusercontent.com/73841520/134240680-c6ae8f65-9484-416d-9a5a-aa32dac44214.png)


# History of Project:
When someone applies to a home credit company to get a loan, estimating the person's inability to pay this loan using the LightGBM algorithm.

# History of Dataset:
* Application table contains 356.255 observations, while it has 123 features.
* Bureau table contains 1.716.428 observations, while it has 17 features.
* Bureau & Balance contains 27.299.925 observations, while it has 3 features.
* Previous table contains 1.670.214 observations, while it has 37 features
* POS Cash Balance table contains 10.001.358 observations, while it has 8 features.
* Installments Payments table contains 13.605.401 observations, while it has 8 features.
* Credit Card Balance table contains 3.840.312 observations, while it has 23 features.

 ### Application 61, Bureau and Bureau Balance 28, Previous Application 25, Posh Cash and Installments Payments 8, and 15 features for Credit Card Balance were produced.
 ### In total, 138 new features were produced.
 ### Working with a total of 848 variables in 7 different data sets, the model estimated with 0.793296 AUC score and 0.79821 Kaggle score.
 ### This Kaggle score ranks 125th out of 7176 competitors.

## Application_{train|test}.csv

This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
Static data for all applications. One row represents one loan in our data sample.

## Bureau.csv
All client's previous credits provided by other financial institutions that were reported to Credit Bureau.

## Bureau_Balance.csv
Monthly balances of previous credits in Credit Bureau.
This table has one row for each month of history of every previous credit reported to Credit Bureau

## POS_CASH_Balance.csv
Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample 

## Credit_Card_Balance.csv
Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample 

## Previous_Application.csv
All previous applications for Home Credit loans of clients who have loans in our sample.
There is one row for each previous application related to loans in our data sample.

## Installments_Payments.csv
Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.


 ![Ekran görüntüsü 2021-09-21 222838](https://user-images.githubusercontent.com/73841520/134241835-9f855239-564d-4575-b845-97a58923fd02.png)

![image](https://user-images.githubusercontent.com/73841520/134247558-1ed03762-a5ba-43c1-8851-4a2926ec64a0.png)
![image](https://user-images.githubusercontent.com/73841520/134247599-20d58a3b-3da8-4005-904f-3dd35407ed1f.png)
![image](https://user-images.githubusercontent.com/73841520/134247618-6f45fc55-f12a-4d58-a047-9246a4ef42c6.png)


