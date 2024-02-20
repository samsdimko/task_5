import findspark
findspark.init()


import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import col, count, max, unix_timestamp
from pyspark.sql.window import Window
from pyspark.sql.functions import dense_rank
from pyspark.sql.functions import sum


spark = SparkSession.builder.appName("Spark-PosgreSQL").getOrCreate()


connection_properties = {
    "url": "jdbc:postgresql://localhost:5432/postgres",
    "user": "postgres",
    "password": "asdf",
    "driver": "org.postgresql.Driver"
}


df_category = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "category") \
                .load()

df_film_category = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "film_category") \
                .load()

df_actor = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "actor") \
                .load()

df_film_actor = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "film_actor") \
                .load()

df_film = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "film") \
                .load()

df_inventory = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "inventory") \
                .load()

df_rental = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "rental") \
                .load()

df_customer = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "customer") \
                .load()

df_address = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "address") \
                .load()

df_city = spark.read\
                .format("jdbc") \
                .options(**connection_properties) \
                .option("dbtable", "city") \
                .load()


task_1_df_category = df_category.select(col('name'), col('category_id'))
task_1_df_film_category = df_film_category.select(col('category_id'))
joined_df = task_1_df_film_category.join(task_1_df_category, task_1_df_film_category.category_id == task_1_df_category.category_id, 'inner')

task_1_df_counts = joined_df.groupBy('name').count()
print('1:')
task_1_df_counts.show()


joined_df = df_actor \
  .join(df_film_actor, df_actor.actor_id == df_film_actor.actor_id, "inner") \
  .join(df_film, df_film_actor.film_id == df_film.film_id, "inner") \
  .join(df_inventory, df_film.film_id == df_inventory.film_id, "inner") \
  .join(df_rental, df_inventory.inventory_id == df_rental.inventory_id, "inner")

actor_rental_counts_df = joined_df \
  .select(col("first_name"), col("last_name")) \
  .groupBy("first_name", "last_name") \
  .count().withColumnRenamed("count","count_rents")

top_actors = actor_rental_counts_df.orderBy(col("count_rents").desc()).limit(10)
print('2:')
top_actors.show()


joined_df = df_category \
    .join(df_film_category, df_category.category_id == df_film_category.category_id, "inner") \
    .join(df_film, df_film_category.film_id == df_film.film_id, "inner")

category_cost_df = joined_df.groupBy('name').sum('replacement_cost') \
    .withColumnRenamed("sum(replacement_cost)","cost").orderBy(col('cost').desc()).limit(1)
print('3:')
category_cost_df.show()


joined_df = df_film.join(df_inventory, df_film.film_id == df_inventory.film_id, 'left_anti')
films_not_in_inventory_df = joined_df.select(col('title'))
print('4:')
films_not_in_inventory_df.show()


inner_query_df = df_actor \
  .join(df_film_actor, df_actor.actor_id == df_film_actor.actor_id, "inner") \
  .join(df_film, df_film_actor.film_id == df_film.film_id, "inner") \
  .join(df_film_category, df_film.film_id == df_film_category.film_id, "inner") \
  .join(df_category, df_film_category.category_id == df_category.category_id, "inner") \
  .where(col("name") == "Children")

actor_film_counts = inner_query_df \
  .select(col("first_name"), col("last_name")) \
  .groupBy("first_name", "last_name") \
  .count()

windowSpec = Window.orderBy(col("count").desc())
actor_film_counts_ranked = actor_film_counts.withColumn("dr", dense_rank().over(windowSpec))

top_actors = actor_film_counts_ranked.where(col("dr") < 4).select(col('first_name'), col('last_name'), col('count'))
print('5:')
top_actors.show()


joined_df = df_customer \
  .join(df_address, df_customer.address_id == df_address.address_id, "inner") \
  .join(df_city, df_address.city_id == df_city.city_id, "inner")

city_customer_counts = joined_df \
  .groupBy("city") \
  .agg(
    sum(col("active")).alias("active"),
    count("*").alias("total_customers")
  )

city_customer_counts = city_customer_counts.withColumn(
  "disactive", col("total_customers") - col("active")
).select(col('city'), col('active'), col('disactive'))

print('6:')
city_customer_counts.orderBy(col("disactive").desc()).show()


joined_df = df_category \
  .join(df_film_category, df_category.category_id == df_film_category.category_id, "inner") \
  .join(df_film, df_film.film_id == df_film_category.film_id, "inner") \
  .join(df_inventory, df_film.film_id == df_inventory.film_id, "inner") \
  .join(df_customer, df_customer.store_id == df_inventory.store_id, "inner") \
  .join(df_rental, df_customer.customer_id == df_rental.customer_id, "inner") \
  .join(df_address, df_address.address_id == df_customer.address_id, "inner") \
  .join(df_city, df_city.city_id == df_address.city_id, "inner") \

category_starts_with_a_df = joined_df.where(col("name").like("A%") & col("city").like("%-%"))

category_rental_time_df = category_starts_with_a_df \
  .groupBy("name", "city") \
  .agg(sum(col('return_date') - col("rental_date")).alias('rental_time'))
windowSpec = Window.partitionBy(col('city')).orderBy(col("rental_time").desc())
ranked_category_rental_time_df = category_rental_time_df.withColumn('dr', dense_rank().over(windowSpec))

popular_categories_df = ranked_category_rental_time_df.select("city", "name").where(col("dr") == 1)
print('7:')
popular_categories_df.show(truncate=False)

