package core.learn.trainers

import java.io._

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.regression._
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

import scala.io.Source
import scala.sys.process._

/**
  * Created by avishaylivne on 1/17/16.
  */

object MLlibDriver {
  val usage = """
    Usage: mllibDriver dataPath modelPath modelName [predictionsPath]
        dataPath: Path to data file.
        modelPath: Path to model.
        modelName: The type of the model to be trained.
        labelsPath: Path to file where the indexed labels will be saved.
                    Only used in classification tasks (not in regression tasks).
        predictionsPath: If set, apply existing model on data to predict labels and save predictions in this path.
                         Else (default), train new model on labeled data.
              """

  def main(args: Array[String]): Unit = {
    if (args.length < 4 || args.length > 5) {
      println(usage)
      return
    }
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    if (args.length == 5) {
      predict(dataPath = args(0), modelPath = args(1), modelName = args(2), labelsPath = args(3), predictionsPath = args(4))
    } else {
      train(dataPath = args(0), modelPath = args(1), modelName = args(2), labelsPath = args(3))
    }
  }

  def indexLabels(data: DataFrame, labelsPath: String): DataFrame = {
    val indexer = new StringIndexer().setInputCol("label").setOutputCol("label1")
    val indexerModel = indexer.fit(data)
    val transformedData = indexerModel.transform(data).select("label1", "features").withColumnRenamed("label1", "label")
    val pw = new PrintWriter(new File(labelsPath))
    indexerModel.labels.map(label => pw.println(label.toDouble.toInt))
    pw.close
    transformedData
  }

  def train(dataPath: String, modelPath: String, modelName: String, labelsPath: String) = {
    val conf = new SparkConf().setAppName(s"MLlib train $modelName")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val rawData = sqlContext.createDataFrame(
      sc.textFile(dataPath).map { line =>
        val parts = line.split('\t')
        (parts(parts.length - 2).toDouble, Vectors.dense(parts.slice(0, parts.length - 2).map(_.toDouble)))
      }).toDF("label", "features")

    var data = rawData
    val modelClass = modelName match {
      case "MLlibLogisticRegression" => {
        data = indexLabels(rawData, labelsPath)
        val m = new LogisticRegression()
        m.setRegParam(0.1)
        m
      }
      case "MLlibLinearRegression" => {
        val m = new LinearRegression()
        m.setRegParam(0.1)
        m
      }
      case "MLlibDecisionTreeClassifier" => {
        data = indexLabels(rawData, labelsPath)
        new DecisionTreeClassifier()
      }
      case "MLlibNaiveBayes" => {
        data = indexLabels(rawData, labelsPath)
        new NaiveBayes()
      }
      case "MLlibRandomForestClassifier" => {
        data = indexLabels(rawData, labelsPath)
        new RandomForestClassifier()
      }
      case "MLlibDecisionTreeRegressor" => new DecisionTreeRegressor()
      case "MLlibGBTRegressor" => new GBTRegressor()
      case "MLlibRandomForestRegressor" => new RandomForestRegressor()
      case _ => throw new IllegalArgumentException(s"Model $modelName is not supported")
    }
    val model = modelClass.fit(data)
    val oos = new ObjectOutputStream(new FileOutputStream(modelPath))
    oos.writeObject(model)
    oos.close
  }

  def readIndexedLabels(labelsPath: String) = {
    Source.fromFile(labelsPath).getLines().map(_.toInt).zipWithIndex.map(_.swap).toMap
  }

  def fixPredictions(predictions: DataFrame, labelsPath: String) = {
    val indexToLabel = readIndexedLabels((labelsPath))
    // translates prediction index to prediction label
    def correctLabel = udf[Double, Int] (indexToLabel(_).toDouble)
    // sort probability vector according to mapping from index to labels
    def correctProbability = udf[List[Double], DenseVector] (v => List.tabulate(v.size){i => v(indexToLabel(i))})
    predictions.select("prediction", "probability")
      .withColumn("fixedPrediction", correctLabel(predictions("prediction")))
      .drop("prediction")
      .withColumnRenamed("fixedPrediction", "prediction")
      .withColumn("fixedProbability", correctProbability(predictions("probability")))
      .drop("probability")
      .withColumnRenamed("fixedProbability", "probability")
  }

  def predict(dataPath: String, modelPath: String, modelName: String, labelsPath: String, predictionsPath: String) = {
    val conf = new SparkConf().setAppName(s"MLlib predict $modelName")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val data = sqlContext.createDataFrame(
      sc.textFile(dataPath).map { line =>
        val parts = line.split('\t')
        (0, Vectors.dense(parts.slice(0, parts.length).map(_.toDouble)))
      }).toDF("label", "features")
    val is = new ObjectInputStream(new FileInputStream(modelPath))
    val predictions = modelName match {
      case "MLlibLogisticRegression" => fixPredictions(is.readObject().asInstanceOf[LogisticRegressionModel].transform(data), labelsPath)
      case "MLlibLinearRegression" => is.readObject().asInstanceOf[LinearRegressionModel].transform(data).select("prediction")
      case "MLlibDecisionTreeClassifier" => fixPredictions(is.readObject().asInstanceOf[DecisionTreeClassificationModel].transform(data), labelsPath)
      case "MLlibNaiveBayes" => fixPredictions(is.readObject().asInstanceOf[NaiveBayesModel].transform(data), labelsPath)
      case "MLlibRandomForestClassifier" => fixPredictions(is.readObject().asInstanceOf[RandomForestClassificationModel].transform(data), labelsPath)
      case "MLlibDecisionTreeRegressor" => is.readObject().asInstanceOf[DecisionTreeRegressionModel].transform(data).select("prediction")
      case "MLlibGBTRegressor" => is.readObject().asInstanceOf[GBTRegressionModel].transform(data).select("prediction")
      case "MLlibRandomForestRegressor" => is.readObject().asInstanceOf[RandomForestRegressionModel].transform(data).select("prediction")
      case _ => throw new IllegalArgumentException(s"Model $modelName is not supported")
    }
    is.close()
    predictions.write.mode(SaveMode.Overwrite).json(predictionsPath)
    println(s"Applied $modelName model on ${predictions.count} rows.")
  }
}
