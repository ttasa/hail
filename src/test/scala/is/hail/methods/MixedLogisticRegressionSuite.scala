package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.utils.D_==
import is.hail.variant.Variant
import org.testng.annotations.Test

/**
  * Created by ttasa on 16/08/2017.
  */
class MixedLogisticRegressionSuite  extends SparkSuite {
  val X_fe = FailOps.open_("Test.Xfe.csv")
  val X_re = FailOps.open_("Test.Xre.csv")
  val y = FailOps.open_("Test.y.csv")

    def assertDouble(a: Annotation, value: Double, tol: Double = 1e-6) {
      assert(D_==(a.asInstanceOf[IndexedSeq[Double]].apply(0), value, tol))
    }

    def assertNaN(a: Annotation) {
      assert(a.asInstanceOf[IndexedSeq[Double]].apply(0).isNaN)
    }

    val v1 = Variant("1", 1, "C", "T") // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T") // x = (., 2, ., 2, 0, 0)
    val v3 = Variant("1", 3, "C", "T") // x = (0, ., 1, 1, 1, .)
    val v6 = Variant("1", 6, "C", "T") // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T") // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T") // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T") // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

  def main(args: Array[String]): Unit = {
    //@Test def testWithTwoCov() {
    /**
      val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
        types = Map("Cov1" -> TFloat64, "Cov2" -> TFloat64)).keyBy("Sample")
      val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
        types = Map("Pheno" -> TFloat64), missing = "0").keyBy("Sample")

      val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
        .annotateSamplesTable(covariates, root = "sa.cov")
        .annotateSamplesTable(phenotypes, root = "sa.pheno")
        .linreg(Array("sa.pheno"), Array("sa.cov.Cov1", "sa.cov.Cov2 + 1 - 1"))

      val a = vds.variantsAndAnnotations.collect().toMap

      val qBeta = vds.queryVA("va.linreg.beta")._2
      val qSe = vds.queryVA("va.linreg.se")._2
      val qTstat = vds.queryVA("va.linreg.tstat")._2
      val qPval = vds.queryVA("va.linreg.pval")._2

      /*
      comparing to output of R code:
      y = c(1, 1, 2, 2, 2, 2)
      x = c(0, 1, 0, 0, 0, 1)
      c1 = c(0, 2, 1, -2, -2, 4)
      c2 = c(-1, 3, 5, 0, -4, 3)
      df = data.frame(y, x, c1, c2)
      fit <- lm(y ~ x + c1 + c2, data=df)
      summary(fit)["coefficients"]
      */
*/
    }
}
