create schema assignment_3;
use assignment_3;

select count(id) from loan;
select verification_status from loan;

/*RES00*/
select purpose, count(id)
from loan
where verification_status = "verified"
group by purpose
order by count(id) desc
limit 5;

/*RES01*/
select purpose, (sum(bad_loan)/count(id)) as default_rate
from loan
where verification_status = 'verified'
group by purpose
order by default_rate desc
limit 5;

/*RES02*/
select addr_state, (sum(bad_loan)/count(id)) as default_rate
from loan
where verification_status = 'verified'
group by addr_state
having count(id) > 1000
order by default_rate desc
limit 5;

/*RES03*/
select purpose, avg(annual_inc) as average_income, avg(loan_amnt) as average_loan_amt, (sum(bad_loan)/count(id)) as default_rate
from loan
where verification_status = 'verified'
group by purpose
order by average_loan_amt desc
limit 5;

/*RES04*/
select pred, bad_loan from test_data;

/*RES05*/
select addr_state, avg(annual_inc) as avg_annual_income, avg(dti) as avg_dti from loan 
where verification_status = 'verified' and longest_credit_length > 10
group by addr_state
order by avg_dti desc
limit 5;