package is.hail.methods

import is.hail.{SparkSuite, TestUtils}
import java.io.File

import breeze.linalg.csvread
import breeze.linalg.DenseMatrix
import is.hail.SparkSuite.hc
import is.hail.expr.{TDict, TFloat64}

object FailOps {



  val X_fe = FailOps.open_("Test.Xfe.csv")
  val X_re = FailOps.open_("Test.Xre.csv")
  val y = FailOps.open_("Test.y.csv")
  val phi = 0.007
  val c = 1.0
  val inc_m = false

  def open_(filename: String): DenseMatrix[Double] = {
    val matrix = breeze.linalg.csvread(new File(filename), skipLines = 1)
    matrix
  }

  def main(args: Array[String]): Unit = {

   // val X_ae = hc.importGen("gen.gen","samp.samp") .write("SampleData.vds")
    val SampleData = hc.read("SampleData.vds").annotateGenotypesExpr("g = g.GT.toGenotype()").toVDS
    val annotations = hc.importTable(input = "samp.annot", keyNames = Array("ID_1"), types = Map("pheno1" -> TFloat64, "cov2" -> TFloat64, "cov3" -> TFloat64))
    val vds_result = SampleData.annotateSamplesTable(annotations, root = "sa.phenotypes")
    // val (y, cov, completeSamples) = RegressionUtils.getPhenosCovCompleteSamples(vds_result, Array("sa.phenotypes.pheno1 "), Array("sa.phenotypes.cov2 ", "sa.phenotypes.cov3 "))
    val covExpr = Array("sa.phenotypes.cov2 ", "sa.phenotypes.cov3 ")
    val vds2=vds_result.logmmreg("sa.phenotypes.pheno1 ",covExpr, "global.logmmreg", "va.logmmreg")
    val DT=vds2.queryGlobal("global.logmmreg.beta").asInstanceOf[Tuple2[String,Map[String,Double]]]._2
    val vec=DT.values.map(_.toDouble)
    print(vec)

  }

}
