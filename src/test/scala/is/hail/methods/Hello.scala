package is.hail.methods

import BlockMatrixMultiplyRDD.BlockMatrixIsDistributedMatrix
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._
import breeze.linalg._
import is.hail.SparkSuite.hc
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.apache.spark.mllib.linalg.Vectors.dense
import org.apache.spark.mllib.optimization.{LBFGS, SimpleUpdater}
import splash.optimization.LeastSquaresGradient

/**
  * Created by ttasa on 07/07/2017.
  *
  * Tegevused implementatsioonil
  *
  * ### 1) Data generation (X.re, X.fe, y):
  * ####
  * #### 1.1 Alguses loeme R'st v2lja kirjutatud andmed niisama sisse.
  * #### 2)  Implementation of the actual algorithm
  * #### 3) Data generation (Not needed)
  * ##
  * ---Results
  * #### 3)SGD for lin. solve /LBFGS
  * #### 4)Matrix multiplication..
  *
  * KOLMAP2EV:
  * Integreeri õigetre andme struktuuridega.
  * 5)Integration with Scala/Spark/Hail and other data structures
  *
  * 6)Test suite
  */


object Hello {

  val X_fe = FailOps.open_("Test.Xfe.csv")
  val X_re = FailOps.open_("Test.Xre.csv")
  val y = FailOps.open_("Test.y.csv")
  val phi = 0.007
  val c=1.0
  val inc_m = false

  def main(args: Array[String]): Unit = {

    val M=X_fe.rows
    val P0=diag(DenseVector.fill(M){c})
    val k = y-0.5
    print(inc_m)
    val X: DenseMatrix[Double] = {
      if (inc_m) {
      val tmp: DenseMatrix[Double] = DenseMatrix.tabulate(X_fe.rows, 1) {case(i) => 1.0  }
     // val tmp: DenseMatrix = ones(X_fe.rows,1)
        DenseMatrix.horzcat(tmp,X_fe, X_re)
      } else {
        DenseMatrix.horzcat(X_fe, X_re)
      }
    }
    val P_a=X_re.cols
    val P_b=X_fe.cols
    val P_ab=P_a+P_b


    val m_idc: Int =if (inc_m){
      1
    }else{
      0
    }
    val P_all=P_ab+m_idc
    var PP = DenseMatrix.zeros[Double](P_all,P_all)
    if (inc_m){
      PP(0,0)=c
    }
    val P1 = diag(DenseVector.fill(P_a){phi/M})
    val N=X.rows

    val a_idc=(0+ m_idc to (P_b-1)+ m_idc)
    val b_idc=(0+P_b+m_idc to (P_a-1)+P_b+m_idc)

    val mu=diag(DenseVector.fill(P_all){0.0})
    val normal_ = breeze.stats.distributions.Gaussian(0,2)

    var beta_old=  DenseVector.rand(P_all, normal_)
    var dbm=  DenseVector.rand(P_all, normal_)
    print(dbm)

    PP(a_idc ,a_idc ):=P0
    PP(b_idc ,b_idc ):=P1
    val i_PP: DenseMatrix[Double]=inv(PP)
    val X_t=  X.t
    def toBM(x: DenseMatrix[Double], rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix =
      BlockMatrixIsDistributedMatrix.from(hc.sc, new org.apache.spark.mllib.linalg.DenseMatrix(x.rows, x.cols, x.toArray), rowsPerBlock, colsPerBlock)

    val BM_X_t:BlockMatrix = toBM(X_t,100,100)
    val BM_X:BlockMatrix = toBM(X,100,100)

    val numIterations = 500
    val stepSize = 0.00005
    val numCorrections = 500
    val convergenceTol = 1e-5
    val maxNumIterations = 500
    val regParam = 0
    var dbm34 = Vectors.dense(new Array [Double] (P_all.toInt))
    while (abs(sum(beta_old-dbm))>1E-5){
      println(abs(sum(beta_old-dbm)))

      ///Matrix multiplication

      var psi= (X*dbm).toDenseVector
     var w = ((1.0:/(2.0 :* psi)) :* tanh(psi :/ 2.0)).toArray
   //   var w2 = ((1.0:/(2.0 :* psi)) :* tanh(psi :/ 2.0))

     //var S: DenseMatrix[Double]=X_t * diag(y.toDenseVector) * X
     var firstMultiply= BlockMatrixIsDistributedMatrix.vectorPointwiseMultiplyEveryColumn(w)(BM_X)

      var S=BlockMatrixIsDistributedMatrix.multiply(BM_X_t,firstMultiply).toLocalMatrix()
      var S2 =  new DenseMatrix[Double](S.numRows, S.numCols, S.toArray)
      beta_old:=dbm
      //Solve with SGD

      //Data preparation
      var tmp_B: DenseMatrix[Double]=(X_t*k+i_PP*mu)

      var tmp_A:DenseMatrix[Double]=(S2 + i_PP)
      val it:Int=tmp_B.rows-1
      val xz:Int=tmp_A.cols-1

    //  print( Vectors.dense(tmp_A(::,1).data))

      val ss = (0 to it).map( e => (tmp_B(e,0),
        Vectors.dense( (tmp_A(::,e)).toArray))
        )



val parsedData2=hc.sc.parallelize(ss)

/**
      val (dbm3, loss) = LBFGS.runLBFGS(
        parsedData2,
        new org.apache.spark.mllib.optimization.LeastSquaresGradient(),
        new SimpleUpdater(),
        numCorrections,
        convergenceTol,
        maxNumIterations,
        regParam,
        dbm34)
  */

      val dbm3 = (new splash.optimization.StochasticGradientDescent())
        .setGradient(new LeastSquaresGradient())
        .setNumIterations(5000)
        .setStepSize(0.5)
        .setDataPerIteration(1.0)
        .optimize(parsedData2, dbm34)



      dbm34=dbm3.copy
      println("LBFGS")
      dbm=new DenseVector[Double](dbm3.toArray)
      print(dbm(0 to 3))
    //  dbm=(tmp_A \ tmp_B).toDenseVector

//     dbm=((S2 + i_PP) \ (X_t*k+i_PP*mu)).toDenseVector

    }
print(dbm)
    //PP(b_idc ,b_idc )



  }
}

/** comments


  */