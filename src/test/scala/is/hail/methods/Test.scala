package is.hail.methods

import BlockMatrixMultiplyRDD.BlockMatrixIsDistributedMatrix
import breeze.linalg._
import is.hail.SparkSuite._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import splash.optimization.LeastSquaresGradient

/**
  * Created by ttasa on 14/08/2017.
  */
object Test {

  def main(args: Array[String]): Unit = {

    val BA = hc.sqlContext.read.option("header", "true").option("inferSchema", "true").csv("SolveBA.csv")

    val A = FailOps.open_("SolveA.csv")
    val A_a=new org.apache.spark.mllib.linalg.DenseMatrix(A.rows,
      A.cols,
      A.data)
    val B = FailOps.open_("SolveB.csv")

    val dbm = FailOps.open_("Solvedbm.csv")


   def toBM(x: DenseMatrix[Double], rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix =
      BlockMatrixIsDistributedMatrix.from(hc.sc, new org.apache.spark.mllib.linalg.DenseMatrix(x.rows, x.cols, x.toArray), rowsPerBlock, colsPerBlock)

    val BM_A:BlockMatrix = toBM(A,100,100)
    val BM_B:BlockMatrix = toBM(B,100,100)
    val BM_x=BlockMatrixIsDistributedMatrix.multiply(BM_A,BM_B).toLocalMatrix()

    //  print(((A) \ (B)).toDenseVector(1))
    //  print(dbm)
    //  print(BA.collect().foreach(println))




    //creating features column
    val assembler = new VectorAssembler()
      .setInputCols(Array("A1", "A2", "A3", "A4", "A5"))
      .setOutputCol("features")


    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0)
      .setElasticNetParam(0)
      .setFeaturesCol("features") // setting features column
      .setLabelCol("label")
      .setFitIntercept(false) // setting label column

    //creating pipeline
    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    val lrMode = pipeline.fit(BA)
    val lrModel = lrMode.stages(1).asInstanceOf[LinearRegressionModel]
    //print(rfModel)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
/**
    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  */
    // Fit the model
    //val r = Array.range(1, BA.columns.length)
    val r = Array.range(1, 6)

    val parsedData = BA.rdd.map(row =>

      new LabeledPoint(
        row.getDouble(0),
        Vectors.dense( r.map(num=> row.getDouble(num)))
      )
    ).cache()
    val numIterations = 1000
    val stepSize = 0.000005
    val lrModel2 = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)
    val valuesAndPreds = parsedData.map { point =>
      val prediction = lrModel2.predict(point.features)
      (point.label, prediction)
    }
    print(lrModel2.intercept)
    print(lrModel2.weights)
    val numCorrections = 500
    val convergenceTol = 1e-8
    val maxNumIterations = 500
    val regParam = 0
    val initialWeightsWithIntercept = Vectors.dense(new Array [Double] (5))


    val parsedData2 = BA.rdd.map(row =>
      (
        row.getDouble(0),
        Vectors.dense( r.map(num=> row.getDouble(num)))
      )
    ).cache()
    parsedData2.take(2).foreach(println)

val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
      parsedData2,
      new org.apache.spark.mllib.optimization.LeastSquaresGradient(),
      new SimpleUpdater(),
      numCorrections,
      convergenceTol,
      maxNumIterations,
      regParam,
 initialWeightsWithIntercept)
    println("LBFGS")
println (weightsWithIntercept)

/**
    val convergenceTol2 = 1e-6
val maxNumIterations2=1000
    val (weightsWithIntercept2, loss2) = GradientDescent.runMiniBatchSGD(
parsedData2,
      new LeastSquaresGradient(),
      new SimpleUpdater(),
      0.000005,
      maxNumIterations2,
      0.00001,
      0.6,initialWeightsWithIntercept,convergenceTol2)
    println("MiniBatch")
    print (weightsWithIntercept2)
*/

    val weights = (new splash.optimization.StochasticGradientDescent())
      .setGradient(new LeastSquaresGradient())
      .setNumIterations(2000)
      .setStepSize(0.5)
        .setDataPerIteration(0.8)
      .optimize(parsedData2, initialWeightsWithIntercept)
    print(weights)



    /**
    val trainingSummary2 = lrModel2.summary
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("training Mean Squared Error = " + MSE)


    println(s"Coefficients: ${lrModel2.coefficients} Intercept: ${lrModel2.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel2.summary
    println(s"numIterations: ${trainingSummary2.totalIterations}")
    println(s"objectiveHistory: [${2.objectiveHistory.mkString(",")}]")
    trainingSummary2.residuals.show()
    println(s"RMSE: ${trainingSummary2.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary2.r2}")
    // Fit the model
      */
  }

}
