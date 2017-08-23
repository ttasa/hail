package is.hail.methods

import splash.optimization.LeastSquaresGradient
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.{LBFGS, SimpleUpdater}

/**
  * Created by ttasa on 17/08/2017.
  * This file returns an approximate result for the system of linear equations
  * Currently, two methods can be selected
  * 1) LBFGS optimisation (spark.mllib)
  * 2) Stochastic Gradient Descent (Splash)
  */
object Optimisation {

  /**
    *Optimisation
    */
  val numIterations = 500
  val stepSize = 0.0005
  val numCorrections = 200
  val convergenceTol = 1e-5
  val maxNumIterations = 1000
  val regParam = 0
  val dataFraction =0.9

  def apply(dataPrepOptParallel: RDD[(Double, Vector)], algorithm: String, coefficients: Vector): Vector = algorithm match {

    case "LBFGS" => {
      val coefficientUpdate = LBFGS.runLBFGS(
        dataPrepOptParallel,
        new org.apache.spark.mllib.optimization.LeastSquaresGradient(),
        new SimpleUpdater(),
        numCorrections,
        convergenceTol,
        maxNumIterations,
        regParam,
        coefficients)._1
      coefficientUpdate
    }

    case "SGD" => {

      val coefficientUpdate = (new splash.optimization.StochasticGradientDescent())
        .setGradient(new LeastSquaresGradient())
        .setNumIterations(numIterations)
        .setStepSize(stepSize)
        .setDataPerIteration(dataFraction)
        .optimize(dataPrepOptParallel, coefficients)
      coefficientUpdate
    }
    case default => {
      val str_ = "No such optimisation method " + default + " is available! \n Current options: SGD, LBFGS, Direct!"
      throw new Exception(str_)
    }

  }
}


