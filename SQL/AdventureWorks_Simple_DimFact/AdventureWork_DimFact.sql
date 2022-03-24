USE AdventureWork_DW;

-- #########################################
-- DECLARE DIM TABLES
-- #########################################

CREATE TABLE DIM_PRODUCT (
	PRODUCT_KEY INT IDENTITY (1,1),
    PRODUCT_ID INT NOT NULL,
    PRODUCT_NAME NVARCHAR(50) NOT NULL,
    PRODUCT_COLOR NVARCHAR(15) NULL,
    PRODUCT_CATEGORY_NAME NVARCHAR(50) NOT NULL,
    CONSTRAINT "DIM_PRODUCT.PK" PRIMARY KEY NONCLUSTERED (PRODUCT_KEY)
)

GO

CREATE UNIQUE CLUSTERED INDEX "DIM_PRODUCT.PRODUCT_ID" ON DIM_PRODUCT(PRODUCT_ID)

GO

CREATE TABLE DIM_CUSTOMER (
    CUSTOMER_KEY INT IDENTITY(1,1),
    CUSTOMER_ID INT NOT NULL,
    CUSTOMER_NAME VARCHAR(120) NULL,
    CONSTRAINT "DIM_CUSTOMER.PK" PRIMARY KEY NONCLUSTERED (CUSTOMER_KEY)
)

GO

CREATE UNIQUE CLUSTERED INDEX "DIM_CUSTOMER.CUSTOMER_ID" ON DIM_CUSTOMER(CUSTOMER_ID)

GO

CREATE TABLE DIM_REGION (
    REGION_KEY INT IDENTITY(1,1),
    REGION_ID INT NOT NULL,
    REGION_NAME NVARCHAR(50) NOT NULL,
    REGION_COUNTRY NVARCHAR(50) NOT NULL,
    CONSTRAINT "DIM_REGION.PK" PRIMARY KEY NONCLUSTERED (REGION_KEY)
)

GO 

CREATE UNIQUE CLUSTERED INDEX "DIM_REGION.REGION_ID" ON DIM_REGION(REGION_ID)

GO

CREATE TABLE DIM_DATE(
    DATE_KEY INT,
    CALENDAR_DATE DATETIME NOT NULL,
    CALENDAR_YEAR INT NOT NULL,
    CALENDAR_MONTH INT NOT NULL,
    CALENDAR_QUARTER INT NOT NULL,
    CALENDAR_WEEK INT NOT NULL,
    CONSTRAINT "DIM_DATE.PK" PRIMARY KEY NONCLUSTERED (DATE_KEY)
)

GO

CREATE UNIQUE CLUSTERED INDEX "DATE_DIM.CALENDAR_DATE" ON DIM_DATE(CALENDAR_DATE)

GO

CREATE TABLE DIM_STORE (
	STORE_KEY INT IDENTITY(1,1),
	STORE_ID INT NOT NULL,
	STORE_NAME VARCHAR(150) NOT NULL
	CONSTRAINT "DIM_STORE.PK" PRIMARY KEY NONCLUSTERED (STORE_KEY)
)


GO

CREATE UNIQUE CLUSTERED INDEX "DIM_STORE.STORE_ID" ON DIM_STORE(STORE_ID)

-- #########################################
-- DECLARE FACT TABLE
-- #########################################


CREATE TABLE FACT_SALES (
    FACT_SALE_ID INT IDENTITY(1,1),
    PRODUCT_KEY INT NOT NULL,
    CUSTOMER_KEY INT NOT NULL,
    REGION_KEY INT NOT NULL,
    DATE_KEY INT NOT NULL,
    SALES_VOLUMES MONEY NOT NULL,
    SHIPPED_UNITS INT NOT NULL,
    CONSTRAINT "FACT_SALES.PK" PRIMARY KEY NONCLUSTERED (FACT_SALE_ID),
    CONSTRAINT "FACT_SALES.REF_DIM_PRODUCT" 
        FOREIGN KEY (PRODUCT_KEY)
        REFERENCES DIM_PRODUCT(PRODUCT_KEY),
    CONSTRAINT "FACT_SALES.REF_DIM_CUSTOMER" 
        FOREIGN KEY (CUSTOMER_KEY)
        REFERENCES DIM_CUSTOMER(CUSTOMER_KEY),
    CONSTRAINT "FACT_SALES.REF_DIM_REGION" 
        FOREIGN KEY (REGION_KEY)
        REFERENCES DIM_REGION(REGION_KEY),
    CONSTRAINT "FACT_SALES.REF_DIM_DATE" 
        FOREIGN KEY (DATE_KEY)
        REFERENCES DIM_DATE(DATE_KEY)
)

GO

CREATE INDEX "FACT_SALES.SALESFACT_ID" ON FACT_SALES(
    PRODUCT_KEY,
    CUSTOMER_KEY,
    REGION_KEY,
    DATE_KEY
)