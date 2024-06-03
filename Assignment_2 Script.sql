use assignment_2;

select count(*) from assignment_2.flights;

select airline, count(*) as flight_count
from flights
group by airline
order by flight_count desc;

describe flights;

/* RES01*/
drop table if exists RES01;
CREATE TABLE RES01 (SELECT airline,
    COUNT(*) AS flight_count,
    AVG(arr_delay) AS avg_arr_delay FROM
    flights
GROUP BY airline
ORDER BY avg_arr_delay DESC);

SELECT 
    *
FROM
    res01;

/* RES02*/
drop table if exists RES02;
CREATE TABLE RES02 (SELECT DATE_FORMAT(fl_date, '%M') AS month_name,
    airline,
    COUNT(*) AS flight_count,
    AVG(arr_delay) AS avg_arr_delay FROM
    flights
GROUP BY airline , month_name
ORDER BY month_name ASC , avg_arr_delay DESC);

SELECT 
    *
FROM
    res02;
    
/* RES03*/
drop table if exists RES03;
CREATE TABLE RES03 (SELECT airline,
    AVG(CASE
        WHEN DATE_FORMAT(fl_date, '%M') = 'january' THEN arr_delay
        ELSE 0
    END) AS january_mean_arr_delay,
    AVG(CASE
        WHEN DATE_FORMAT(fl_date, '%M') = 'may' THEN arr_delay
        ELSE 0
    END) AS may_mean_arr_delay,
    (AVG(CASE
        WHEN DATE_FORMAT(fl_date, '%M') = 'january' THEN arr_delay
        ELSE 0
    END) - AVG(CASE
        WHEN DATE_FORMAT(fl_date, '%M') = 'may' THEN arr_delay
        ELSE 0
    END)) AS january_vs_may_change FROM
    flights
GROUP BY airline
ORDER BY january_vs_may_change ASC);

SELECT 
    *
FROM
    res03;
    
/* RES04*/
drop table if exists RES04;
CREATE TABLE RES04 (SELECT DATE_FORMAT(fl_date, '%W') AS day_of_week,
    AVG(dep_delay) AS avg_dep_delay,
    AVG(arr_delay) AS avg_arr_delay,
    COUNT(*) AS flight_count FROM
    flights
GROUP BY day_of_week
ORDER BY avg_dep_delay DESC);

SELECT 
    *
FROM
    res04;
    
/* RES05*/
drop table if exists RES05;
CREATE TABLE RES05 (SELECT CASE
        WHEN
            DATE_FORMAT(fl_date, '%w') = 0
                OR DATE_FORMAT(fl_date, '%w') = 6
        THEN
            'Weekend'
        ELSE 'Weekday'
    END AS week_day_indicator,
    AVG(dep_delay) AS avg_dep_delay,
    AVG(arr_delay) AS avg_arr_delay,
    COUNT(*) AS flight_count FROM
    flights
GROUP BY week_day_indicator
ORDER BY avg_dep_delay DESC);
)

SELECT 
    *
FROM
    res05;
    
/* RES06*/
drop table if exists RES06;
CREATE TABLE RES06 (SELECT airline,
    AVG(distance) AS mean_dist,
    MIN(distance) AS min_dist,
    MAX(distance) AS max_dist FROM
    flights
GROUP BY airline
ORDER BY mean_dist ASC);

SELECT 
    *
FROM
    res06;
    
/* RES07*/
drop table if exists RES07;
CREATE TABLE RES07 (SELECT airline,
    AVG(air_time) AS mean_time,
    MIN(air_time) AS min_time,
    MAX(air_time) AS max_time FROM
    flights
GROUP BY airline
ORDER BY mean_time ASC);

SELECT 
    *
FROM
    res07;
    
/* RES08*/
drop table if exists RES08;
CREATE TABLE RES08 (SELECT airline,
    origin_state_nm,
    COUNT(*) AS flight_count,
    AVG(arr_delay) AS avg_dep_delay FROM
    flights
WHERE
    origin_state_nm = 'Florida'
GROUP BY airline
ORDER BY avg_dep_delay ASC);

SELECT 
    *
FROM
    res08;
    
/* RES09*/
drop table if exists RES09;
CREATE TABLE RES09 (SELECT origin_city_name,
    dest_city_name,
    MAX(air_time) AS max_airtime,
    MAX(air_time) / 60 AS max_airtime_hrs FROM
    flights
GROUP BY origin_city_name , dest_city_name
ORDER BY max_airtime DESC
LIMIT 5);

SELECT 
    *
FROM
    res09;
    
/* RES10*/
drop table if exists RES10;
CREATE TABLE RES10 (SELECT origin_city_name,
    dest_city_name,
    AVG(air_time) AS mean_airtime,
    AVG(air_time) / 60 AS mean_airtime_hrs,
    COUNT(*) AS flight_count FROM
    flights
GROUP BY origin_city_name , dest_city_name
HAVING AVG(arr_delay) > 15
    AND flight_count > 10
ORDER BY mean_airtime DESC
LIMIT 5);

SELECT 
    *
FROM
    res10;
    
/* RES11*/
drop table if exists RES11;
CREATE TABLE RES11 (SELECT origin_city_name,
    dest_city_name,
    AVG(arr_delay) AS mean_arr_delay,
    COUNT(*) AS flight_count FROM
    flights
GROUP BY origin_city_name , dest_city_name
HAVING mean_arr_delay < -10
    AND flight_count > 20
ORDER BY mean_arr_delay ASC
LIMIT 5);

SELECT 
    *
FROM
    res11;
    
/* RES12*/
drop table if exists RES12;
CREATE TABLE RES12 (SELECT origin_city_name,
    dest_city_name,
    AVG(arr_delay) AS mean_arr_delay,
    COUNT(*) AS flight_count FROM
    flights
WHERE
    weather_delay > 0
GROUP BY origin_city_name , dest_city_name
HAVING mean_arr_delay > 15 AND flight_count > 2
ORDER BY mean_arr_delay ASC
LIMIT 5);

SELECT 
    *
FROM
    res12;
    
/* RES13*/
drop table if exists RES13;
CREATE TABLE RES13 (SELECT airline,
    dest_city_name,
    AVG(carrier_delay) / 60 AS mean_carrier_delay_hrs FROM
    flights
WHERE
    arr_delay > 30
GROUP BY airline , dest_city_name
HAVING AVG(carrier_delay) / 60 > 2
ORDER BY mean_carrier_delay_hrs DESC) LIMIT 5;

SELECT 
    *
FROM
    res13;
    
/* RES14*/
drop table if exists RES14;
CREATE TABLE RES14 (SELECT dest_city_name,
    (CASE
        WHEN COUNT(*) > 1000 THEN 'high traffic'
        WHEN COUNT(*) < 1000 THEN 'medium traffic'
        WHEN COUNT(*) < 700 THEN 'low traffic'
    END) AS traffic_level,
    COUNT(*) AS flight_count FROM
    flights
GROUP BY dest_city_name
ORDER BY flight_count DESC
LIMIT 5);

SELECT 
    *
FROM
    res14;
    
/* RES15*/
drop table if exists RES15;
CREATE TABLE RES15 (SELECT airline,
    origin_city_name,
    dest_city_name,
    dep_delay / 60 AS dep_delay_hrs FROM
    flights
WHERE
    dep_delay / 60 < (SELECT 
            AVG(dep_delay) / 60
        FROM
            flights)
ORDER BY dep_delay_hrs ASC) LIMIT 5;

SELECT 
    *
FROM
    res15;