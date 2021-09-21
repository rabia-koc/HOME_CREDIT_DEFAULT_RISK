# HOME CREDIT DEFAULT RISK

![Ekran görüntüsü 2021-09-21 231044](https://user-images.githubusercontent.com/73841520/134240680-c6ae8f65-9484-416d-9a5a-aa32dac44214.png)

## application_{train|test}.csv

This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
Static data for all applications. One row represents one loan in our data sample.

## bureau.csv
All client's previous credits provided by other financial institutions that were reported to Credit Bureau.

## bureau_balance.csv
Monthly balances of previous credits in Credit Bureau.
This table has one row for each month of history of every previous credit reported to Credit Bureau

## POS_CASH_balance.csv
Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample 

## credit_card_balance.csv
Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample 

## previous_application.csv
All previous applications for Home Credit loans of clients who have loans in our sample.
There is one row for each previous application related to loans in our data sample.

## installments_payments.csv
Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.



