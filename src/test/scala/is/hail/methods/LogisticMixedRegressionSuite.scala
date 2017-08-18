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


  def assertDouble(a: Annotation, value: Double, tol: Double = 1e-6) {
    assert(D_==(a.asInstanceOf[Double], value, tol))
  }
  def assertVector(a: Annotation, value: Double , tol: Double = 1e-6) {
    assert(D_==(a.asInstanceOf[Double], value, tol))
  }

  @Test def logmmMainTest {
  //  val X_ae = hc.importGen("gen.gen","samp.samp").write("SampleData.vds")
    val SampleData = hc.read("SampleData.vds").annotateGenotypesExpr("g = g.GT.toGenotype()").toVDS
    val annotations = hc.importTable(input = "samp.annot", keyNames = Array("ID_1"), types = Map("pheno1" -> TFloat64, "cov2" -> TFloat64, "cov3" -> TFloat64))
    val vds_result = SampleData.annotateSamplesTable(annotations, root = "sa.phenotypes")
    val covExpr = Array("sa.phenotypes.cov2 ", "sa.phenotypes.cov3 ")
    val vds2=vds_result.logmmreg("sa.phenotypes.pheno1 ",covExpr, "global.logmmreg", "va.logmmreg")
    val DT=vds2.queryGlobal("global.logmmreg.beta").asInstanceOf[Tuple2[String,Map[String,Double]]]._2
    val vec=DT.values.map(_.toDouble).asInstanceOf[List[Double]]
    assert(D_==(vec(0),0.5061129016885633))
    assert(D_==(vec(1),0.204382200315554))
    assert(D_==(vec(2),-0.4217789905591546))
  }


}
