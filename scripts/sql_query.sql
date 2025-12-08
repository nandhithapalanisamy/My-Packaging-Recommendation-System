--sql query
--creating database

CREATE DATABASE infosys_database;

--creating table for material_data

CREATE TABLE material_data (
    id SERIAL PRIMARY KEY NOT NULL,
    material_type VARCHAR(150)NOT NULL,
    strength VARCHAR(50),
    weight_capacity REAL,
    biodegradability_score NUMERIC(5,2),
    co2_emission_score NUMERIC(5,2),
    recyclability_percent REAL,
    compostable VARCHAR(10),
    water_resistance VARCHAR(50)
);

--importing material_data from csv file

COPY material_data(material_type, strength, weight_capacity,
                   biodegradability_score, co2_emission,
                   recyclability, compostable, water_resistance)
FROM 'C:/Users/nandh/Downloads/material_data.csv'
DELIMITER ','
CSV HEADER;

--schema validation

--  Validate data types and column names

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'material_data';

--  Check for missing important values

SELECT *
FROM material_data
WHERE 
    material_type IS NULL OR
    strength IS NULL OR
    weight_capacity IS NULL OR
    biodegradability_score IS NULL OR
    co2_emission IS NULL OR
    recyclability IS NULL;

-- Validate numeric ranges

SELECT *
FROM material_data
WHERE 
    weight_capacity < 0 OR
    biodegradability_score NOT BETWEEN 0 AND 100 OR
    co2_emission < 0 OR
    recyclability NOT BETWEEN 0 AND 100;

--  Validate compostable values

SELECT *
FROM material_data
WHERE compostable NOT IN ('yes', 'no');

-- Validate strength consistency

SELECT *
FROM material
WHERE strength NOT IN ('low', 'low-medium', 'medium', 'high');

-- Validate water resistance levels

SELECT *
FROM material
WHERE water_resistance NOT IN ('low', 'medium', 'high');

--creating table for product_data

CREATE TABLE product_data (
    productID SERIAL PRIMARY KEY NOT NULL,
    productCategory VARCHAR(100),
    productName VARCHAR(150),
    productWeight REAL,
    width REAL,
    height REAL
);
--importing product_data from csv file

COPY product_data(productid,productcategory,productname,productweight,width,height)
FROM 'C:/Users/nandh/Downloads/product_data.csv'
DELIMITER ','
CSV HEADER;

--schema validation

--  Validate data types and column names

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'product_data';

--  Check for missing important values

SELECT *
FROM product_data
WHERE productCategory IS NULL 
   OR productName IS NULL 
   OR productWeight IS NULL
   OR width IS NULL
   OR height IS NULL;

-- Validate numeric ranges

SELECT *
FROM product_data
WHERE productWeight <= 0
   OR width <= 0
   OR height <= 0;

