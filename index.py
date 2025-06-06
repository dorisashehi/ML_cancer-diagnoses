from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnull, isnan
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# -----------------------------------------
# 1. Initialize Spark session
# -----------------------------------------
spark = SparkSession.builder \
    .appName("CancerDiagnosisOptimized") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")                  # Reduce log verbosity

# -----------------------------------------
# 2. Load dataset
# -----------------------------------------
print("Loading data...")
df = spark.read.csv("project3_data.csv", header=True, inferSchema=True).cache()  # Read and cache data

# -----------------------------------------
# 3. Drop unnecessary columns
# -----------------------------------------
if 'id' in df.columns:
    df = df.drop("id")  # Drop identifier column which isn't useful for modeling

# -----------------------------------------
# 4. Cast features to numeric (double)
# -----------------------------------------
feature_cols = [c for c in df.columns if c not in ["diagnosis", "label"]]
df = df.select(*(col(c).cast("double").alias(c) if c in feature_cols else col(c) for c in df.columns))

# -----------------------------------------
# 5. Drop columns that are completely null
# -----------------------------------------
null_counts = df.select([count(when(isnull(c) | isnan(c), c)).alias(c) for c in feature_cols]).first().asDict()
all_null_cols = [c for c in feature_cols if null_counts[c] == df.count()]
if all_null_cols:
    df = df.drop(*all_null_cols)
    feature_cols = [c for c in feature_cols if c not in all_null_cols]

# -----------------------------------------
# 6. Impute missing values in partially-null columns
# -----------------------------------------
partial_null_cols = [c for c in feature_cols if 0 < null_counts[c] < df.count()]
if partial_null_cols:
    imputer = Imputer(inputCols=partial_null_cols, outputCols=partial_null_cols, strategy="mean")
    df = imputer.fit(df).transform(df)

# -----------------------------------------
# 7. Encode labels and assemble features
# -----------------------------------------
df = StringIndexer(inputCol="diagnosis", outputCol="label").fit(df).transform(df)  # Encode target labels
df = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(df).select("label", "features").cache()

# -----------------------------------------
# 8. Split data into training and testing sets
# -----------------------------------------
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)  # 80/20 split with fixed seed for reproducibility

# -----------------------------------------
# 9. Set up evaluator for model performance
# -----------------------------------------
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# -----------------------------------------
# 10. Train Random Forest with cross-validation
# -----------------------------------------
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
rf_param_grid = ParamGridBuilder().addGrid(rf.numTrees, [50]).build()  # Test 50 trees for speed

rf_cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=rf_param_grid,
    evaluator=evaluator.setMetricName("accuracy"),
    numFolds=3  # 3-fold cross-validation to reduce computation
)
rf_model = rf_cv.fit(train_data)               # Train model
rf_preds = rf_model.transform(test_data)       # Predict on test data

# -----------------------------------------
# 11. Train Gradient Boosted Trees with cross-validation
# -----------------------------------------
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=50)  # Limit iterations
gbt_param_grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3]).build()     # Test one depth value

gbt_cv = CrossValidator(
    estimator=gbt,
    estimatorParamMaps=gbt_param_grid,
    evaluator=evaluator.setMetricName("accuracy"),
    numFolds=3
)
gbt_model = gbt_cv.fit(train_data)              # Train model
gbt_preds = gbt_model.transform(test_data)      # Predict on test data

# -----------------------------------------
# 12. Evaluate both models using multiple metrics
# -----------------------------------------
metrics = {
    "Accuracy": "accuracy",
    "F1 Score": "f1",
    "Precision": "weightedPrecision",
    "Recall": "weightedRecall"
}

print("\nModel Evaluation:")
for metric_name, metric_code in metrics.items():
    rf_score = evaluator.setMetricName(metric_code).evaluate(rf_preds)
    gbt_score = evaluator.setMetricName(metric_code).evaluate(gbt_preds)
    print(f"{metric_name:<12} | Random Forest: {rf_score:.4f} | Gradient Boosting: {gbt_score:.4f}")

# -----------------------------------------
# 13. Stop the Spark session
# -----------------------------------------
spark.stop()
