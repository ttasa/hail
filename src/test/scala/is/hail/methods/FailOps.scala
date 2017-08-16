package is.hail.methods

import java.io.File
import breeze.linalg.csvread
import breeze.linalg.DenseMatrix

object FailOps {

  def open_ (filename : String): DenseMatrix[Double] = {
    val matrix = breeze.linalg.csvread(new File(filename),skipLines = 1)
  matrix
  }

}
