package is.hail.methods


import is.hail.SparkSuite
import is.hail.stats.RegressionUtils
import is.hail.annotations.Annotation
import is.hail.utils.D_==
import is.hail.expr.TFloat64
import org.testng.annotations.Test

/**
  * Created by ttasa on 17/08/2017.
  */

class LogisticMixedRegressionSuite extends SparkSuite {
  // val X_ae = hc.importGen("src/test/resources/LogitMM.gen","src/test/resources/LogitMM.samp").write("src/test/resources/LogitMM.vds")
  val SampleData = hc.read("src/test/resources/LogitMM.vds").annotateGenotypesExpr("g = g.GT.toGenotype()").toVDS


  val annotations = hc.importTable(input = "src/test/resources/LogitMM.annot1", keyNames = Array("ID_1"), types = Map("pheno1" -> TFloat64, "cov2" -> TFloat64, "cov3" -> TFloat64))
  val vds_result = SampleData.annotateSamplesTable(annotations, root = "sa.phenotypes")
  val covExpr = Array("sa.phenotypes.cov2 ", "sa.phenotypes.cov3 ")

  @Test def logmmLBFGSTest {

    val vds2 = vds_result.logmmreg("sa.phenotypes.pheno1 ", covExpr, "global.logmmreg", "va.logmmreg", optMethod = "LBFGS", c = 1.0, phi = 0.007)
    val DT = vds2.queryGlobal("global.logmmreg.beta").asInstanceOf[Tuple2[String, Map[String, Double]]]._2
    val vec = DT.values.map(_.toDouble).asInstanceOf[List[Double]]
    assert(D_==(vec(0), 0.50611, tolerance = 1e-4))
    assert(D_==(vec(1), 0.20438, tolerance = 1e-4))
    assert(D_==(vec(2), -0.42177, tolerance = 1e-4))
  }

  @Test def logmmDirectTest {

    val vds2 = vds_result.logmmreg("sa.phenotypes.pheno1 ", covExpr, "global.logmmreg", "va.logmmreg", optMethod = "Direct", c = 1.0, phi = 0.007)
    val DT = vds2.queryGlobal("global.logmmreg.beta").asInstanceOf[Tuple2[String, Map[String, Double]]]._2
    val vec = DT.values.map(_.toDouble).asInstanceOf[List[Double]]
    assert(D_==(vec(0), 0.50611, tolerance = 1e-4))
    assert(D_==(vec(1), 0.20438, tolerance = 1e-4))
    assert(D_==(vec(2), -0.42177, tolerance = 1e-4))
  }

}
