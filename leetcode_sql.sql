-- 一些最重要的 SQL 命令
-- SELECT - 从数据库中提取数据
-- UPDATE - 更新数据库中的数据
-- DELETE - 从数据库中删除数据
-- INSERT INTO - 向数据库中插入新数据
-- CREATE DATABASE - 创建新数据库
-- ALTER DATABASE - 修改数据库
-- CREATE TABLE - 创建新表
-- ALTER TABLE - 变更（改变）数据库表
-- DROP TABLE - 删除表
-- CREATE INDEX - 创建索引（搜索键）
-- DROP INDEX - 删除索引

-- 620. 有趣的电影 (Not Boring Movies)
SELECT
    * 
FROM 
    cinema 
WHERE
    id % 2 = 1 AND description != 'boring'
ORDER BY 
    rating DESC
;

-- 620. 有趣的电影 (Not Boring Movies)
SELECT 
    * 
FROM 
    cinema 
WHERE 
    MOD(id,2) = 1 AND description <> 'boring' 
ORDER BY 
    rating DESC
;


-- 595. 大的国家 (Big Countries)
SELECT
    name , population , area 
FROM 
    World
WHERE 
    area >= 3000000 OR population >= 25000000
;


-- 1757. 可回收且低脂的产品 (Recyclable and Low Fat Products)
SELECT 
    product_id
FROM 
    Products
WHERE
    low_fats = 'Y' AND recyclable = 'Y'
;

-- 1741. 查找每个员工花费的总时间 (Find Total Time Spent by Each Employee)
SELECT
    event_day AS day , emp_id , SUM(out_time - in_time) AS total_time
FROM
    Employees
GROUP BY
    event_day,emp_id
;

-- 1965. 丢失信息的雇员 (Employees With Missing Information)
SELECT
    employee_id
FROM
    (
        SELECT employee_id FROM employees
        UNION ALL
        SELECT employee_id FROM salaries
    ) 
    AS t
GROUP BY
    employee_id
HAVING
    COUNT(employee_id) = 1
ORDER BY
    employee_id ASC
;

-- 175. 组合两个表 (Combine Two Tables)
SELECT
    Person.firstName , Person.lastName ,Address.city,Address.state
FROM 
    Person
LEFT JOIN
    Address
ON
    Person.personId = Address.personId
;

-- 182. 查找重复的电子邮箱 (Duplicate Emails)
SELECT
    Email
FROM
    Person
GROUP BY
    Email
HAVING
    COUNT(Email) > 1
;

-- 1587. 银行账户概要 II (Bank Account Summary II)
SELECT
    name , SUM(Transactions.amount) AS balance
FROM
    Users
LEFT JOIN
    Transactions
GROUP BY
    Users.account
HAVING
    balance > 10000
;



