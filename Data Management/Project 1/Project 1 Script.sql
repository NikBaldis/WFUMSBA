/*Create Schema*/
drop schema if exists project1;
create schema project1;

/*Create Employee Table*/
use project1;
drop table if exists employee
CREATE TABLE employee (
    employee_id INT NOT NULL AUTO_INCREMENT,
    employee_name VARCHAR(64),
    Employee_title VARCHAR(64),
    Employee_dept VARCHAR(64),
    full_or_part_time CHAR(1),
    salary_or_hourly VARCHAR(10),
    typical_hours DECIMAL(5 , 2 ),
    annual_salary DECIMAL(10 , 2 ),
    hourly_rate DECIMAL(5 , 2 ),
    PRIMARY KEY (employee_id)
);

describe employee;

/*Insert Employee Data*/
INSERT INTO employee
(employee_id,
employee_name,
employee_title,
employee_dept,
full_or_part_time,
salary_or_hourly,
typical_hours,
annual_salary,
hourly_rate
)
VALUES 
(null,'AARON,JEFFERY M','SERGEANT','POLICE','F','Salary',null,111444.00,null);

INSERT INTO employee
(employee_name,
employee_title,
employee_dept,
full_or_part_time,
salary_or_hourly,
annual_salary
)
VALUES 
('AARON,KARI','POLICE OFFICER (ASSIGNED AS DETECTIVE)','POLICE','F','Salary',94122.00);

INSERT INTO employee
(employee_name,
employee_title,
employee_dept,
full_or_part_time,
salary_or_hourly,
annual_salary
)
VALUES 
('AARON, KIMBERLEI R','CHIEF CONTRACT EXPEDITER','DAIS','F','Salary',118608.00);

INSERT INTO employee
(employee_name,
employee_title,
employee_dept,
full_or_part_time,
salary_or_hourly,
annual_salary
)
Values
('ABAD JR, VICENTE M','CIVIL ENGINEER IV','WATER MGMNT','F','Salary',117072.00);

INSERT INTO employee
(employee_name,
employee_title,
employee_dept,
full_or_part_time,
salary_or_hourly,
typical_hours,
hourly_rate
)
Values
('ABARCA, EMMANUEL','CONCRETE LABORER','TRANSPORTN','F','Hourly',40,44.4);

INSERT INTO employee
(employee_name,
employee_title,
employee_dept,
full_or_part_time,
salary_or_hourly,
annual_salary
)
Values
('ABARCA, FRANCES J','POLICE OFFICER','POLICE','F','Salary',68616.00);

INSERT INTO employee
(employee_name,
employee_title,
employee_dept,
full_or_part_time,
salary_or_hourly,
typical_hours,
hourly_rate
)
Values
('ABASCAL, REECE E','TRAFFIC CONTROL AIDE-HOURLY','OEMC','P','Hourly',20,19.86);

INSERT INTO employee
(employee_name,
employee_title,
employee_dept,
full_or_part_time,
salary_or_hourly,
typical_hours,
hourly_rate
)
Values
('ABBATACOLA, ROBERT J','ELECTRICAL MECHANIC','AVIATION','F','Hourly',40,50);

select * 
from employee;

/*SQL Queries
Question 1*/
SELECT 
    *
FROM
    employee
WHERE
    salary_or_hourly = 'Salary'
        AND annual_salary < 100000

/*Question 2*/
SELECT 
    *,
    (typical_hours * hourly_rate * 50) AS estimated_annual_salary
FROM
    employee
WHERE
    salary_or_hourly = 'Hourly'
ORDER BY estimated_annual_salary DESC;

/* Question 3*/
SELECT 
    *
FROM
    employee
WHERE
    Employee_title LIKE '%OFF%';

/* Task 2
Create and Load NYC_Applications, create nyc_applications_prep*/
describe nyc_applications_prep;
SELECT 
    *
FROM
    nyc_applications_prep
LIMIT 10;

/*Question 1 */
Drop table if exists Res01
CREATE TABLE Res01 (SELECT restaurant_name,
    business_address,
    borough,
    sidewalk_dimensions_area,
    qualify_alcohol FROM
    nyc_applications_prep
WHERE
    borough = 'Manhattan'
        AND seating_interest_sidewalk = 'sidewalk'
ORDER BY sidewalk_dimensions_area DESC);
SELECT 
    *
FROM
    Res01
LIMIT 10;

select * from nyc_applications_prep
limit 10;
/* Question 2*/
Drop table if exists Res02
CREATE TABLE Res02 (SELECT restaurant_name,
    borough,
    business_address,
    sidewalk_dimensions_area,
    qualify_alcohol FROM
    nyc_applications_prep
WHERE
    borough = 'Brooklyn'
        AND qualify_alcohol = 'yes'
ORDER BY sidewalk_dimensions_area DESC);
SELECT 
    *
FROM
    Res02
LIMIT 10;

/* Question 3*/
Drop table if exists Res03
CREATE TABLE Res03 (SELECT restaurant_name,
    business_address,
    borough,
    sidewalk_dimensions_area,
    qualify_alcohol FROM
    nyc_applications_prep
WHERE
    qualify_alcohol = 'yes'
        AND restaurant_name like '%pizza%'
ORDER BY sidewalk_dimensions_area DESC);
SELECT 
    *
FROM
    Res03
LIMIT 10;

/* Question 4*/
Drop table if exists Res04
CREATE TABLE Res04 (SELECT restaurant_name,
    business_address,
    sidewalk_dimensions_area,
    borough,
    qualify_alcohol FROM
    nyc_applications_prep
WHERE
    qualify_alcohol = 'yes'
        AND borough = 'Brooklyn'
        AND sidewalk_dimensions_area > 0
ORDER BY sidewalk_dimensions_area ASC);
SELECT 
    *
FROM
    Res04
LIMIT 10;

/* Question 5*/
Drop table if exists Res05
CREATE TABLE Res05 (SELECT restaurant_name,
    business_address,
    sidewalk_dimensions_area,
    borough,
    qualify_alcohol FROM
    nyc_applications_prep
WHERE
    seating_interest_sidewalk = 'sidewalk'
        AND borough = 'Queens'
        AND sidewalk_dimensions_area > 0
ORDER BY sidewalk_dimensions_area ASC);
SELECT 
    *
FROM
    Res05
LIMIT 10;

/* Question 6*/
Drop table if exists Res06
CREATE TABLE Res06 (SELECT restaurant_name,
    business_address,
    borough,
    sidewalk_dimensions_area,
    qualify_alcohol FROM
    nyc_applications_prep
WHERE
    restaurant_name like 'Thai%'
        AND borough = 'Manhattan'
        AND qualify_alcohol = 'yes'
ORDER BY sidewalk_dimensions_area DESC);
SELECT 
    *
FROM
    Res06
LIMIT 10;

/* Question 7*/
Drop table if exists Res07
CREATE TABLE Res07 (SELECT restaurant_name,
    business_address,
    sidewalk_dimensions_area,
    borough,
    roadway_dimensions_area,
    qualify_alcohol,
    (sidewalk_dimensions_area + roadway_dimensions_area) AS total_outside_area FROM
    nyc_applications_prep
ORDER BY total_outside_area DESC);
SELECT restaurant_name,
    business_address,
    sidewalk_dimensions_area,
    borough,
    roadway_dimensions_area,
    qualify_alcohol
FROM
    Res07
LIMIT 5;

/* Question 8*/
Drop table if exists Res08
CREATE TABLE Res08 (SELECT restaurant_name,
    business_address,
    borough,
    sidewalk_dimensions_area,
    roadway_dimensions_area FROM
    nyc_applications_prep
WHERE
    seating_interest_sidewalk = 'both'
        AND borough = 'Brooklyn'
ORDER BY sidewalk_dimensions_area ASC);

SELECT 
    *
FROM
    Res08
WHERE
    sidewalk_dimensions_area = 0
        OR roadway_dimensions_area = 0
ORDER BY sidewalk_dimensions_area ASC
LIMIT 10;
