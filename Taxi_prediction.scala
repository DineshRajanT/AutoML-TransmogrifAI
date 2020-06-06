

package com.salesforce.hw

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataReaders
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import java.io.Serializable
import com.salesforce.op.stages.impl.regression.RegressionModelSelector
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry.{OpGBTRegressor, OpDecisionTreeRegressor}
import com.salesforce.op.stages.impl.tuning.{DataCutter, DataSplitter}

case class Taxi_prediction(
pickup_lat: String,
pickup_long: String,
date_col: String,
day_of_the_week: String,
OnHour: Option[Int],
cust_density: Int
)

object Taxi_prediction {
  /**
   * Run this from the command line with
   * ./gradlew sparkSubmit -Dmain=com.salesforce.hw.Taxi_prediction -Dargs=/full/path/to/csv/file
   */
def main(args: Array[String]): Unit = {
if (args.isEmpty) {
println("You need to pass in the CSV file path as an argument")
sys.exit(1)
}
val csvFilePath = args(0)
// val mon = args(1)
println(s"Using user-supplied CSV file path: $csvFilePath")


implicit val spark = SparkSession.builder.config(new SparkConf()).getOrCreate()
import spark.implicits._

/*
implicit val srEncoder = Encoders.product[Taxi_prediction]

object Extractors extends Serializable {
class pickup_lat_extractor extends Function[Taxi_prediction,Text] with Serializable {
def apply(i:Taxi_prediction): Text = i.pickup_lat.toText
}
class pickup_long_extractor extends Function[Taxi_prediction,Text] with Serializable {
def apply(o:Taxi_prediction): Text = o.pickup_long.toText
}

class date_col_extractor extends Function[Taxi_prediction,Text] with Serializable {
def apply(o:Taxi_prediction): Text = o.date_col.toText
}
class day_of_the_week_extractor extends Function[Taxi_prediction,Text] with Serializable {
def apply(o:Taxi_prediction): Text = o.day_of_the_week.toText
}
class OnHour_extractor extends Function[Taxi_prediction,Real] with Serializable {
def apply(o:Taxi_prediction): Real = o.OnHour.toReal
}
class cust_density_extractor extends Function[Taxi_prediction,RealNN] with Serializable {
def apply(o:Taxi_prediction): RealNN = o.cust_density.toRealNN
}

}

import Extractors._
*/

val pickup_lat = FeatureBuilder.Text[Taxi_prediction].extract(_.pickup_lat.toText).asPredictor
val pickup_long = FeatureBuilder.Text[Taxi_prediction].extract(_.pickup_long.toText).asPredictor
val date_col = FeatureBuilder.Text[Taxi_prediction].extract(_.date_col.toText).asPredictor
val day_of_the_week = FeatureBuilder.Text[Taxi_prediction].extract(_.day_of_the_week.toText).asPredictor
val OnHour = FeatureBuilder.Real[Taxi_prediction].extract(_.OnHour.toReal).asPredictor
val cust_density = FeatureBuilder.RealNN[Taxi_prediction].extract(_.cust_density.toRealNN).asResponse



// val dataReader = List("40.743", "-73.996", "2015/01/07", "Wednesday", 22, 8)


// val dataReader = new DataReader[Taxi_prediction](
// pickup_lat: "40.743",
// pickup_long: "-73.996",
// date_col: "2015/01/07",
// day_of_the_week: "Wednesday",
// OnHour: 22,
// cust_density: 8)

val dataReader = DataReaders.Simple.csvCase[Taxi_prediction](path = Option(csvFilePath), key = _.date_col.toString)
println("Showing DataReader here!!!")
// println(dataReader)

val features = Seq(pickup_lat, pickup_long, date_col, day_of_the_week, OnHour).transmogrify()
val randomSeed = 42L
val splitter = DataSplitter(seed = randomSeed)

val prediction1 = RegressionModelSelector
.withCrossValidation(
dataSplitter = Some(splitter), seed = randomSeed,
modelTypesToUse = Seq(OpGBTRegressor, OpDecisionTreeRegressor)
).setInput(cust_density, features).getOutput()

val evaluator = Evaluators.Regression().setLabelCol(cust_density).setPredictionCol(prediction1)
val workflow = new OpWorkflow("abc").setResultFeatures(prediction1, cust_density).setReader(dataReader)


val workflowModel = workflow.loadModel(path = "/home/whirldata/Documents/moving/from_server/savings/model1")
println(s"Model summary:\n${workflowModel.summaryPretty()}")


val testDataReader = DataReaders.Simple.csvCase[Taxi_prediction](path = Option(csvFilePath))
val scoredTestData = workflowModel.setReader(testDataReader).score()

val dfScoreAndEvaluate = workflowModel.scoreAndEvaluate(evaluator)
val dfScore = dfScoreAndEvaluate._1.
withColumnRenamed("input_features", "predicted_total_amount")
val dfEvaluate = dfScoreAndEvaluate._2

// prediction column
val dfScoreMod = dfScore.rdd.map(x => x(2).toString.split("->")(1).dropRight(1).dropRight(1))
dfScoreMod.foreach(println)
dfScoreMod.saveAsTextFile("./output_new/Feb_data-15.txt")

println("Scoring the model")
val (scores, metrics) = workflowModel.scoreAndEvaluate(evaluator = evaluator)

println("Metrics:\n" + metrics)

spark.stop()
}
}
