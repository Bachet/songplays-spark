import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek, from_unixtime
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType


config = configparser.ConfigParser()
config.read("dl.cfg")

os.environ["AWS_ACCESS_KEY_ID"] = config["AWS"]["ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = config["AWS"]["SECRET_ACCESS_KEY"]


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data, schema):
    # get filepath to song data file
    song_data = input_data + "song-data/A/A/*/"
    
    # read song data file
    df = spark.read.schema(schema).json(song_data)

    # extract columns to create songs table
    songs_table = df.select(col("song_id"), col("title"), col("artist_id"), col("year"), col("duration"))
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").mode("overwrite").parquet(output_data + "songs_table/")

    # extract columns to create artists table
    artists_table = df.select(
        col("artist_id"), col("artist_name").alias("name"), col("artist_location").alias("location"),
        col("artist_latitude").alias("latitude"), col("artist_longitude").alias("longitude")
    ).distinct()
    
    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet(output_data + "artists_table/")

    return df


def process_log_data(spark, input_data, output_data, schema, song_df):
    # get filepath to log data file
    log_data = input_data + "log-data/*/*/"

    # read log data file
    df = spark.read.schema(schema).json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table    
    users_table = df.select(
        col("userId").alias("user_id"), col("firstName").alias("first_name"),
        col("lastName").alias("last_name"), col("gender"), col("level")
    ).distinct()
    
    # write users table to parquet files
    users_table.write.mode("overwrite").parquet(output_data + "users_table/")

    # create timestamp column from original timestamp column
    df = df.withColumn("timestamp", from_unixtime(col("ts")/1000))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda z: datetime.utcfromtimestamp(z / 1000).strftime("%Y-%m-%d %H:%M:%S"), StringType())
    df = df.withColumn("datetime", get_datetime(col("ts")))
    
    # extract columns to create time table
    time_table = df.select(
        col("timestamp").alias("start_time"), hour(col("datetime")).alias("hour"),
        dayofmonth(col("datetime")).alias("day"), weekofyear(col("datetime")).alias("week"),
        month(col("datetime")).alias("month"), year(col("datetime")).alias("year"),
        dayofweek(col("datetime")).alias("weekday")
    )
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").mode("overwrite").parquet(output_data + "time_table/")

    # read in song data to use for songplays table
    songplays_df = df.join(
        song_df,
        on=(df.song == song_df.title) & (df.artist == song_df.artist_name) & (df.length == song_df.duration),
        how="left"
    )

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = songplays_df.select(
        monotonically_increasing_id().alias("songplay_id"), col("timestamp").alias("start_time"),
        col("userId").alias("user_id"), col("level"), col("song_id"), col("artist_id"),
        col("sessionId").alias("session_id"), col("location"), col("userAgent").alias("user_agent"),
        year(col("datetime")).alias("year"), month(col("datetime")).alias("month")
    )

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").mode("overwrite").parquet(output_data + "songplays_table/")


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://uda-songplays-spark/"

    song_data_schema = StructType([
        StructField("artist_id", StringType(), True),
        StructField("artist_name", StringType(), True),
        StructField("artist_location", StringType(), True),
        StructField("artist_latitude", DoubleType(), True),
        StructField("artist_longitude", DoubleType(), True),
        StructField("duration", DoubleType(), True),
        StructField("num_songs", LongType(), True),
        StructField("song_id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("year", IntegerType(), True)
    ])

    log_data_schema = StructType([
        StructField("artist", StringType(), True),
        StructField("auth", StringType(), True),
        StructField("firstName", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("itemInSession", LongType(), True),
        StructField("lastName", StringType(), True),
        StructField("length", DoubleType(), True),
        StructField("level", StringType(), True),
        StructField("location", StringType(), True),
        StructField("method", StringType(), True),
        StructField("page", StringType(), True),
        StructField("registration", DoubleType(), True),
        StructField("sessionId", LongType(), True),
        StructField("song", StringType(), True),
        StructField("status", IntegerType(), True),
        StructField("ts", LongType(), True),
        StructField("userAgent", StringType(), True),
        StructField("userId", StringType(), True)
    ])
    
    song_df = process_song_data(spark, input_data, output_data, song_data_schema)
    process_log_data(spark, input_data, output_data, log_data_schema, song_df)


if __name__ == "__main__":
    main()
