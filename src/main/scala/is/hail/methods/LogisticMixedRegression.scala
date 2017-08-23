package is.hail.methods

import BlockMatrixMultiplyRDD.BlockMatrixIsDistributedMatrix
import breeze.linalg._
import breeze.numerics.{abs, tanh}
import is.hail.annotations.Annotation
import is.hail.expr.{Parser, TDict, TFloat64, TString, TStruct, Type}
import is.hail.stats.{RegressionUtils, ToNormalizedDenseMatrix}
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.BlockMatrix

/**
  * Created by ttasa on 16/08/2017.
  */

object LogisticMixedRegression {

  /**
    * beta : vector of null model coefficients
    * phi : genetic additive variance component
    * c : diagonal covariance of fixed effects
    */

  val schema: Type = TStruct(("beta", TDict(TString, TFloat64)), ("phi", TFloat64), ("c", TFloat64))

  def toBM(x: DenseMatrix[Double], rowsPerBlock: Int, colsPerBlock: Int, vds_result: VariantDataset): BlockMatrix =
    BlockMatrixIsDistributedMatrix.from(vds_result.sparkContext, new org.apache.spark.mllib.linalg.DenseMatrix(x.rows, x.cols, x.toArray), rowsPerBlock, colsPerBlock)


  /**
    * Parameters
    * yExpr : feature name of dependent variable
    * covExpr : String array of covariate names
    * rootGA : Global annotation path name
    * rootVA : Variant annotation path name
    * runAssoc : if TRUE: runs GWAS per variant based on null model
    * phi : genetic additive variance component
    * c : diagonal covariance of fixed effects
    * optMethod : LBFGS / SGD
    */

  def apply(vds_result: VariantDataset,
            yExpr: String,
            covExpr: Array[String],
            rootGA: String,
            rootVA: String,
            runAssoc: Boolean = false,
            phi: Double = 0.007,
            c: Double = 1,
            optMethod: String = "LBFGS"
           ): VariantDataset = {

    /**
      * Setup of parameters for the EM algorithm
      */
    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds_result, yExpr, covExpr)


    val XKinship = ToNormalizedDenseMatrix(vds_result)
    val incIntercept = false

    val covRows = cov.rows
    val P0 = diag(DenseVector.fill(covRows) {
      c
    })

    val k = (y - 0.5).toDenseMatrix.t

    val X: DenseMatrix[Double] = {
      if (incIntercept) {
        val intercept: DenseMatrix[Double] = DenseMatrix.tabulate(cov.rows, 1) { case (i) => 1.0 }
        // val tmp: DenseMatrix = ones(X_fe.rows,1)
        DenseMatrix.horzcat(intercept, cov, XKinship)
      } else {
        DenseMatrix.horzcat(cov, XKinship)
      }
    }
    val P_a = XKinship.cols
    val P_b = cov.cols
    val P_ab = P_a + P_b
    val m_idc: Int = if (incIntercept) {
      1
    } else {
      0
    }
    val P_all = P_ab + m_idc
    var PP = DenseMatrix.zeros[Double](P_all, P_all)
    if (incIntercept) {
      PP(0, 0) = c
    }
    val P1 = diag(DenseVector.fill(P_a) {
      phi / covRows
    })
    val N = X.rows

    val a_idc = (0 + m_idc to (P_b - 1) + m_idc)
    val b_idc = (0 + P_b + m_idc to (P_a - 1) + P_b + m_idc)

    val mu = diag(DenseVector.fill(P_all) {
      0.0
    })
    val initNormal = breeze.stats.distributions.Gaussian(0, 2)

    var beta_old = DenseVector.rand(P_all, initNormal)
    var dbm = DenseVector.rand(P_all, initNormal)
    PP(a_idc, a_idc) := P0
    PP(b_idc, b_idc) := P1
    val i_PP: DenseMatrix[Double] = inv(PP)
    val X_t = X.t

    val BM_X_t: BlockMatrix = toBM(X_t, 1024, 1024, vds_result)
    val BM_X: BlockMatrix = toBM(X, 1024, 1024, vds_result)

    var coefficients = Vectors.dense(new Array[Double](P_all.toInt))

    /** Implementation of EM algorithm
      */

    while ((abs(sum(beta_old - dbm)) > 1E-5)) {
      println(abs(sum(beta_old - dbm)))
      var psi = (X * dbm).toDenseVector
      // Expected value of w from PG distribution

      var w = ((1.0 :/ (2.0 :* psi)) :* tanh(psi :/ 2.0)).toArray

      var firstMultiply = BlockMatrixIsDistributedMatrix.vectorPointwiseMultiplyEveryColumn(w)(BM_X)

      var S = BlockMatrixIsDistributedMatrix.multiply(BM_X_t, firstMultiply).toLocalMatrix()

      var S2 = new DenseMatrix[Double](S.numRows, S.numCols, S.toArray)
      beta_old := dbm

      /** Components for solving linear equations */
      var systemColumn: DenseMatrix[Double] = (X_t * k + i_PP * mu)
      var systemDesign: DenseMatrix[Double] = (S2 + i_PP)
      val it: Int = systemColumn.rows - 1
      if (optMethod != "Direct")
      {
        val dataPrepOptimise = (0 to it).map(index => (systemColumn(index, 0),
          Vectors.dense((systemDesign(::, index)).toArray))
        )
        /** Solve linear equations with approximate methods*/
        val dataPrepOptParallel = vds_result.sparkContext.parallelize(dataPrepOptimise)
        val coefficientUpdate = Optimisation(dataPrepOptParallel, optMethod, coefficients)
        coefficients = coefficientUpdate.copy
        /** Update coefficient values */
        dbm = new DenseVector[Double](coefficientUpdate.toArray)
      }
      else
      {
        /** Solve linear equations with a direct solve*/
        // dbm = (tmp_A \ tmp_B).toDenseVector
        dbm = (systemDesign \ systemColumn).toDenseVector
      }
      print(dbm(0 to 3))
    }
    //Siia
    /** Annotate the Variant Dataset with null model parameter estimates  */
    val pathVA = Parser.parseAnnotationRoot(rootVA, Annotation.VARIANT_HEAD)
    Parser.validateAnnotationRoot(rootGA, Annotation.GLOBAL_HEAD)

    val covNames = "intercept" +: covExpr
    val globalBetaMap = covNames.zip(dbm.toArray).toMap

    val vds1 = vds_result.annotateGlobal(
      Annotation(globalBetaMap, phi, c),
      schema, rootGA)
    /** Returns a VariantDataset with fixed and random effect parameter global annotations  */

    if (runAssoc) {
      /** Need to implement likelihood ratio test based on predictions from fixed effect betas
        * and then annotate each variant with all relevant statistics (t, beta, se, ...)
        * Currently outputs just the null model result */


      vds1
    } else {
      vds1
    }
  }
}