USE AdventureWork_DW;

INSERT INTO DBO.DIM_CUSTOMER (CUSTOMER_ID, CUSTOMER_TYPE)
SELECT CUSTOMERID, AccountNumber
FROM ADVENTUREWORKS2017.SALES.CUSTOMER;

GO

INSERT INTO DBO.DIM_REGION (REGION_ID, REGION_NAME, REGION_COUNTRY)
SELECT TERRITORYID, NAME, COUNTRYREGIONCODE
FROM ADVENTUREWORKS2017.SALES.SALESTERRITORY;

GO

INSERT INTO DBO.DIM_PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRODUCT_COLOR, PRODUCT_CATEGORY_NAME)
SELECT
    P.PRODUCTID
    , P.NAME AS PRODUCT_NAME
    , P.COLOR AS PRODUCT_COLOR
    , S.NAME AS PRODUCT_CATEGORY_NAME
FROM ADVENTUREWORKS2017.PRODUCTION.PRODUCT P
INNER JOIN ADVENTUREWORKS2017.PRODUCTION.PRODUCTSUBCATEGORY S
    ON P.PRODUCTSUBCATEGORYID = S.PRODUCTSUBCATEGORYID;

GO

DECLARE @START_DATE DATETIME;
DECLARE @END_DATE DATETIME;

SET @START_DATE = '1/1/2011';
SET @END_DATE = '1/1/2015';

WHILE (@START_DATE <= @END_DATE)
    BEGIN
        INSERT INTO DBO.DIM_DATE (DATE_KEY, CALENDAR_DATE, CALENDAR_YEAR, CALENDAR_QUARTER, CALENDAR_MONTH, CALENDAR_WEEK)
        SELECT
			CONVERT(VARCHAR(8), @START_DATE, 112) AS DATE_KEY
            , @START_DATE CALENDAR_DATE
            , DATEPART(YYYY, @START_DATE) AS CALENDAR_YEAR
            , DATEPART(Q, @START_DATE) AS CALENDAR_QUARTER
            , DATEPART(M, @START_DATE) AS CALENDAR_MONTH
            , DATEPART(WK, @START_DATE) AS CALENDAR_WEEK
        SET @START_DATE = @START_DATE + 1
    END;

