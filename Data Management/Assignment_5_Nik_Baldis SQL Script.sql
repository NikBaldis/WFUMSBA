create schema assignment_5;	
use assignment_5;

select * from customer_prep
limit 10;

select count(*) from customer_prep;

select * from full_customer
limit 10;

/*Question 1*/
select credit_score, EVENT_LABEL from full_customer;


/*Question 2*/
select customer_tenure, balance_inqury_count, email_age, balance_current_amt, current_customer, EVENT_LABEL, label_legit from full_customer;

/*Question 3*/
select label_fraud, EVENT_LABEL from full_customer;

/*Question 4*/
select EVENT_LABEL, label_predict, label_fraud from full_customer
where label_fraud >= 0.5;

/*Question 5*/
