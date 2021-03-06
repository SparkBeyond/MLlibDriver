package core.learn.trainers

import java.io._

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.regression._
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.{DataFrame, SQLContext, SaveMode}
import org.apache.spark.sql.functions._
import org.json4s.{DefaultFormats, _}
import org.json4s.jackson.JsonMethods._

import scala.io.Source
/**
  * Created by avishaylivne on 1/17/16.
  */

object MLlibDriver {
  val usage = """
    Usage: mllibDriver "train" dataPath modelPath modelName labelsPath isClassification modelParams
					or
           mllibDriver "predict" dataPath modelPath modelName labelsPath isClassification predictionsPath
        trainOrPredict: If "train", train a new model on labeled data. Otherwise apply an existing model on data to
                        predict labels.

        dataPath: Path to data file.

        modelPath: Path to model.

        modelName: The type of the model to be trained.

        labelsPath: Path to file where the indexed labels will be saved.
                    Only used in classification tasks (not in regression tasks).

				isClassification: Whether the target label is categorical (classification) or numeric (regression).

        The last arguments should be either modelParams (if training) or predictionsPath (if predicting)

				modelParams: Map[String,Any] encoded as JSON string with model parameters the model should use.

        predictionsPath: If set, apply existing model on data to predict labels and save predictions in this path.
                         If omitted, train new model on labeled data.

              """

  def main(args: Array[String]): Unit = {
    if (args.length != 7) {
      println(usage)
      return
    }
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
		val isClassification = args(5).toLowerCase == "true"
		if (args(0) == "train") {
      train(dataPath=args(1), modelPath=args(2), modelName=args(3), labelsPath=args(4),
				isClassification=isClassification, modelParamsString=args(6))
    } else {
      predict(dataPath=args(1), modelPath=args(2), modelName=args(3), labelsPath=args(4),
				isClassification=isClassification, predictionsPath=args(6))
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

  //  private def getDeepLearningConfig(): MultiLayerConfiguration = {
  //    new NeuralNetConfiguration.Builder()
  //      .seed(1L)
  //      .iterations(1)
  //      .learningRate(1e-6f)
  //      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
  //      .momentum(0.9)
  //      .constrainGradientToUnitNorm(true)
  //      .useDropConnect(true)
  //      .list(2)
  //      .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
  //        .nIn(4).nOut(3).weightInit(WeightInit.XAVIER).updater(Updater.ADAGRAD).activation("sigmoid").dropOut(0.5)
  //        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
  //      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
  //        .nIn(3).nOut(3).weightInit(WeightInit.XAVIER).updater(Updater.ADAGRAD).activation("softmax").dropOut(0.5).build())
  //      .build()
  //  }

  def train(dataPath: String, modelPath: String, modelName: String, labelsPath: String, isClassification: Boolean,
						modelParamsString: String) = {
    val conf = new SparkConf().setAppName(s"MLlib train $modelName")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    val rawData = sqlContext.createDataFrame(
      sc.textFile(dataPath).map { line =>
        val parts = line.split('\t')
        (parts(parts.length - 2).toDouble, parts(parts.length - 2).toDouble, Vectors.dense(parts.slice(0, parts.length - 2).map(_.toDouble)))
      }).toDF("label", "weight", "features")

    var data = rawData
    //    if (modelName == "MLlibDeepLearningClassifier") {
    //      val model = new SparkDl4jMultiLayer(sc, getDeepLearningConfig())
    //      model.fit(data, 10)
    //      val oos = new ObjectOutputStream(new FileOutputStream(modelPath))
    //      oos.writeObject(model)
    //      oos.close
    //    } else {
		implicit val formats = DefaultFormats
		val modelParams = parse(modelParamsString).values.asInstanceOf[Map[String,Any]]
    val modelBuilder = modelName match {
      case "MLlibLinearModel" if isClassification => {
        data = indexLabels(rawData, labelsPath)
        val m = new LogisticRegression()
        m.setRegParam(0.1)
        m
      }
      case "MLlibLinearModel" if !isClassification => {
        val m = new LinearRegression()
        m.setRegParam(0.1)
        m
      }
      case "MLlibDecisionTree" if isClassification => {
        data = indexLabels(rawData, labelsPath)
        new DecisionTreeClassifier()
      }
      case "MLlibNaiveBayes" if isClassification => {
        data = indexLabels(rawData, labelsPath)
        new NaiveBayes()
      }
      case "MLlibRandomForest" if isClassification => {
        data = indexLabels(rawData, labelsPath)
        new RandomForestClassifier()
      }
      case "MLlibDecisionTree" if !isClassification => new DecisionTreeRegressor()
      case "MLlibGBTRegressor" if !isClassification => new GBTRegressor()
      case "MLlibRandomForest" if !isClassification => new RandomForestRegressor()
      case _ => throw new IllegalArgumentException(s"Model $modelName is not supported")
    }
    modelParams.foreach{x =>
      if (modelBuilder.hasParam(x._1)) {
        //TODO(avishay): find a better way to force BigInts into Ints
        val value = modelBuilder.getParam(x._1).getClass.toString match {
          case p if p.endsWith("IntParam") => x._2.asInstanceOf[BigInt].toInt
          case p if p.endsWith("LongParam") => x._2.asInstanceOf[BigInt].toLong
          case p if p.endsWith("DoubleParam") => x._2.asInstanceOf[Double]
          case p if p.endsWith("param.Param") => x._2.asInstanceOf[String]
          case x => throw new Exception(s"Unsupported param type $x")
        }
        modelBuilder.set(modelBuilder.getParam(x._1), value)
      }
    }
    val model = modelBuilder.fit(data)
    val oos = new ObjectOutputStream(new FileOutputStream(modelPath))
    oos.writeObject(model)
    oos.close
  }
  //  }

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

  def predict(dataPath: String, modelPath: String, modelName: String, labelsPath: String, isClassification: Boolean,
							predictionsPath: String) = {
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
      case "MLlibLinearModel" if isClassification => fixPredictions(is.readObject().asInstanceOf[LogisticRegressionModel].transform(data), labelsPath)
      case "MLlibLinearModel" if !isClassification => is.readObject().asInstanceOf[LinearRegressionModel].transform(data).select("prediction")
      case "MLlibDecisionTree" if isClassification => fixPredictions(is.readObject().asInstanceOf[DecisionTreeClassificationModel].transform(data), labelsPath)
			case "MLlibDecisionTree" if !isClassification => is.readObject().asInstanceOf[DecisionTreeRegressionModel].transform(data).select("prediction")
      case "MLlibNaiveBayes" if isClassification => fixPredictions(is.readObject().asInstanceOf[NaiveBayesModel].transform(data), labelsPath)
			case "MLlibRandomForest" if isClassification => fixPredictions(is.readObject().asInstanceOf[RandomForestClassificationModel].transform(data), labelsPath)
			case "MLlibRandomForest" if !isClassification => is.readObject().asInstanceOf[RandomForestRegressionModel].transform(data).select("prediction")
      case "MLlibGBTRegressor" if !isClassification => is.readObject().asInstanceOf[GBTRegressionModel].transform(data).select("prediction")
      case _ => throw new IllegalArgumentException(s"Model $modelName is not supported")
    }
    is.close()
    predictions.coalesce(1).write.mode(SaveMode.Overwrite).json(predictionsPath)
    println(s"Applied $modelName model on ${predictions.count} rows.")
  }
}
