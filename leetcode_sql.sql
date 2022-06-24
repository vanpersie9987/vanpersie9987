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

