# import 
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
spark = SparkSession.builder.appName('sample').getOrCreate()

# reading csv 
df = spark.read.csv('penguins.csv', header = True, inferSchema = True)

# like .shape in pandas
print(df.count(), len(df.columns))

# like .info in pandas
df.printSchema()

# like .head() in pandas
df.show(5)

# select multiple columns
df[['island', 'mass']].show(3)

# filtering
df[df['species']].isin(['Chinstrap', 'Gentoo']).show(5)
df[df['species'].rlike('G.')].show(5) # regex
df[df['flipper'].between(225,229)].show(5)
df[df['mass'].isNull()].show(5)
df[(df['mass'] < 3400) & (df['sex'] == 'Male')].show(5)

# sorting
df.orderBy('mass', ascending = False).show(5)
df.orderBy(['mass', 'flipper', ascending = [True, False]]).show(5)

# select distinct n count distinct
df.select('species').distinct().show() # select
df.select('species').distinct().count()

# sum, mean, min, max
df.agg({'flipper':'mean'}).show()
df.agg({'flipper':'sum'}).show()
df.agg(F.min('flipper'), F.max('flipper')).show()


# groupby
df.groupby('species') \ 
    .agg(
        sum('flipper').alias('sum_flipper'),
        avg('flipper').alias('avg_flipper'),
        min('flipper').alias('min_flipper'),
        max('flipper').alias('max_flipper')
    ) \
    .where(col('sum_flipper') >= 1000) \
    .show(truncate = False)

# user define function
from pyspark.sql.functions import udf

@udf(returnType = FloatType())
def add_cols(x, y):
    return x+y

@udf(returnType = StringType)
def switchGender(x)
    if x == 'Male':
        return 'Female'
    return 'Male'

df = (df
    .withColumn('SumFlipperMass', add_cols(df['Flipper'], df['Mass']))
    .withColumn('SwitchGender', switchGender(df['Sex']))
    )
