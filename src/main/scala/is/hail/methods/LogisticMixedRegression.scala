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

  val schema: Type = TStruct(("beta", TDict(TString, TFloat64)), ("phi", TFloat64), ("c", TFloat64))

  def toBM(x: DenseMatrix[Double], rowsPerBlock: Int, colsPerBlock: Int,vds_result : VariantDataset): BlockMatrix =
    BlockMatrixIsDistributedMatrix.from(vds_result.sparkContext, new org.apache.spark.mllib.linalg.DenseMatrix(x.rows, x.cols, x.toArray), rowsPerBlock, colsPerBlock)

  def apply(vds_result: VariantDataset,
            yExpr: String,
            covExpr: Array[String],
            rootGA: String,
            rootVA: String,
            runAssoc: Boolean = false,
            phi: Double = 0.007,
            c: Double = 1
           ): VariantDataset = {

    //This will go into TestSuite Class
    //
    ///Yhy aren't samples automatically annotated when gen files are read in...
    //
    //  val X_ae = hc.importGen("gen.gen","samp.samp") .write("SampleData.vds")
  //  val SampleData = hc.read("XX.vds").annotateGenotypesExpr("g = g.GT.toGenotype()").toVDS
  //  val annotations = hc.importTable(input = "samp.annot", keyNames = Array("ID_1"), types = Map("pheno1" -> TFloat64, "cov2" -> TFloat64, "cov3" -> TFloat64))
   // val vds_result = SampleData.annotateSamplesTable(annotations, root = "sa.phenotypes")
   // val (y, cov, completeSamples) = RegressionUtils.getPhenosCovCompleteSamples(vds_result, Array("sa.phenotypes.pheno1 "), Array("sa.phenotypes.cov2 ", "sa.phenotypes.cov3 "))


    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds_result, yExpr,covExpr)

    //Selle asemel on mul BreezeMatrixit vaja.//Sain

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

    val numIterations = 500
    val stepSize = 0.00005
    val numCorrections = 500
    val convergenceTol = 1e-5
    val maxNumIterations = 500
    val regParam = 0
    var coefficients = Vectors.dense(new Array[Double](P_all.toInt))


    while (abs(sum(beta_old - dbm)) > 1E-5) {
      println(abs(sum(beta_old - dbm)))
      var psi = (X * dbm).toDenseVector
      var w = ((1.0 :/ (2.0 :* psi)) :* tanh(psi :/ 2.0)).toArray

      var firstMultiply = BlockMatrixIsDistributedMatrix.vectorPointwiseMultiplyEveryColumn(w)(BM_X)

      var S = BlockMatrixIsDistributedMatrix.multiply(BM_X_t, firstMultiply).toLocalMatrix()

      var S2 = new DenseMatrix[Double](S.numRows, S.numCols, S.toArray)
      beta_old := dbm
      // Solve
      //with SGD

      // Data preparation

      var systemColumn: DenseMatrix[Double] = (X_t * k + i_PP * mu)
      var systemDesign: DenseMatrix[Double] = (S2 + i_PP)
      val it: Int = systemColumn.rows - 1
      //val xz: Int = systemDesign.cols - 1

      // print (Vectors.dense(tmp_A(::, 1).data))
      val dataPrepOptimise = (0 to it).map(index => (systemColumn(index, 0),
        Vectors.dense((systemDesign(::, index)).toArray))
      )

      val dataPrepOptParallel = vds_result.sparkContext.parallelize(dataPrepOptimise)
      val coefficientUpdate = Optimisation(dataPrepOptParallel, "LBFGS", coefficients)
      coefficients = coefficientUpdate.copy
      dbm = new DenseVector[Double](coefficientUpdate.toArray)
      print(dbm(0 to 3))
      // dbm = (tmp_A \ tmp_B).toDenseVector
      // dbm = ((S2 + i_PP) \ (X_t * k + i_PP * mu)).toDenseVector
    }
    //Siia

    val pathVA = Parser.parseAnnotationRoot(rootVA, Annotation.VARIANT_HEAD)
    Parser.validateAnnotationRoot(rootGA, Annotation.GLOBAL_HEAD)
    //Tee lmmreg'i j2rgi VariantDatasSet function
   // val covExpr: Array[String] = Array("cov2", "cov3")
    val covNames = "intercept" +: covExpr
    val globalBetaMap = covNames.zip(dbm.toArray).toMap
    val vds1 = vds_result.annotateGlobal(
      Annotation(globalBetaMap, phi, c),
      schema, rootGA)

    //runassoc insted of true
    if (false) {

      //Likelihood ratio test based on global beta and fixed effect design matrix
      //Currently replaced with Variant dataset vds2
      vds1

    } else {
      vds1
    }
  }

}