#[cfg(feature = "nalgebra")]
use nalgebralib::{DMatrix, DVector};
use rhai::{Array, Dynamic, EvalAltResult, Position, FLOAT};

/// Wrapper around [`rhai::Array`] representing a matrix.
///
/// This type provides conversions between Rhai arrays and
/// [`nalgebra::DMatrix`].
///
/// # Examples
/// ```
/// use rhai::{Array, Dynamic};
/// use rhai_sci::matrix::RhaiMatrix;
/// let raw: Array = vec![
///     Dynamic::from_array(vec![Dynamic::from_float(1.0), Dynamic::from_float(2.0)]),
///     Dynamic::from_array(vec![Dynamic::from_float(3.0), Dynamic::from_float(4.0)]),
/// ];
/// let matrix = RhaiMatrix::from_array(raw.clone());
/// assert_eq!(matrix.to_array().len(), raw.len());
/// ```
#[derive(Clone, Debug)]
pub struct RhaiMatrix(Array);

impl RhaiMatrix {
    /// Construct a [`RhaiMatrix`] from a [`rhai::Array`].
    #[must_use]
    pub fn from_array(arr: Array) -> Self {
        Self(arr)
    }

    /// Construct a [`RhaiMatrix`] representing a row vector (`1×N`).
    ///
    /// # Examples
    /// ```
    /// use rhai::{Array, Dynamic};
    /// use rhai_sci::matrix::RhaiMatrix;
    /// let data: Array = vec![Dynamic::from_int(1), Dynamic::from_int(2)];
    /// let row = RhaiMatrix::row_vector(data.clone());
    /// assert!(row.as_row().is_some());
    /// ```
    #[must_use]
    pub fn row_vector(data: Array) -> Self {
        Self(vec![Dynamic::from_array(data)])
    }

    /// Construct a [`RhaiMatrix`] representing a column vector (`N×1`).
    ///
    /// # Examples
    /// ```
    /// use rhai::{Array, Dynamic};
    /// use rhai_sci::matrix::RhaiMatrix;
    /// let data: Array = vec![Dynamic::from_int(1), Dynamic::from_int(2)];
    /// let column = RhaiMatrix::column_vector(data.clone());
    /// assert!(column.as_column().is_some());
    /// ```
    #[must_use]
    pub fn column_vector(data: Array) -> Self {
        let rows = data
            .into_iter()
            .map(|v| Dynamic::from_array(vec![v]))
            .collect();
        Self(rows)
    }

    /// Return the matrix as a row vector (`1×N`), reshaping a column vector if necessary.
    ///
    /// Returns `None` if the matrix is not `1×N` or `N×1`.
    ///
    /// # Examples
    /// ```
    /// use rhai::{Array, Dynamic};
    /// use rhai_sci::matrix::RhaiMatrix;
    /// let data: Array = vec![Dynamic::from_int(1), Dynamic::from_int(2)];
    /// let column = RhaiMatrix::column_vector(data.clone());
    /// let row = column.as_row().unwrap();
    /// assert!(row.as_row().is_some());
    /// ```
    #[must_use]
    pub fn as_row(&self) -> Option<Self> {
        let mut arr = self.0.clone();
        let shape = crate::matrix_functions::matrix_size_by_reference(&mut arr.clone());
        if shape.len() == 2 {
            if shape[0].as_int().unwrap() == 1_i64 {
                Some(self.clone())
            } else if shape[1].as_int().unwrap() == 1_i64 {
                let flat = crate::matrix_functions::flatten(&mut arr);
                Some(Self::row_vector(flat))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Return the matrix as a column vector (`N×1`), reshaping a row vector if necessary.
    ///
    /// Returns `None` if the matrix is not `1×N` or `N×1`.
    ///
    /// # Examples
    /// ```
    /// use rhai::{Array, Dynamic};
    /// use rhai_sci::matrix::RhaiMatrix;
    /// let data: Array = vec![Dynamic::from_int(1), Dynamic::from_int(2)];
    /// let row = RhaiMatrix::row_vector(data.clone());
    /// let column = row.as_column().unwrap();
    /// assert!(column.as_column().is_some());
    /// ```
    #[must_use]
    pub fn as_column(&self) -> Option<Self> {
        let mut arr = self.0.clone();
        let shape = crate::matrix_functions::matrix_size_by_reference(&mut arr.clone());
        if shape.len() == 2 {
            if shape[1].as_int().unwrap() == 1_i64 {
                Some(self.clone())
            } else if shape[0].as_int().unwrap() == 1_i64 {
                let flat = crate::matrix_functions::flatten(&mut arr);
                Some(Self::column_vector(flat))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Convert the matrix back into a [`rhai::Array`].
    #[must_use]
    pub fn to_array(self) -> Array {
        self.0
    }

    /// Convert the matrix into a [`nalgebra::DMatrix`].
    ///
    /// # Errors
    /// Returns an error if any element is non-numeric or rows have differing lengths.
    ///
    /// # Panics
    /// Panics if an integer value cannot be represented as `FLOAT`.
    #[cfg(feature = "nalgebra")]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_dmatrix(&self) -> Result<DMatrix<FLOAT>, Box<EvalAltResult>> {
        if self.0.is_empty() {
            return Ok(DMatrix::from_element(0, 0, 0.0));
        }
        let rows = self.0.len();
        let first_row = self.0[0].clone().into_array().map_err(|_| {
            EvalAltResult::ErrorArithmetic(
                "Matrix must contain row arrays".to_string(),
                Position::NONE,
            )
        })?;
        let cols = first_row.len();
        let mut dm = DMatrix::zeros(rows, cols);
        for (i, row_dyn) in self.0.iter().enumerate() {
            let row = row_dyn.clone().into_array().map_err(|_| {
                EvalAltResult::ErrorArithmetic(
                    "Matrix must contain row arrays".to_string(),
                    Position::NONE,
                )
            })?;
            if row.len() != cols {
                return Err(EvalAltResult::ErrorArithmetic(
                    "Matrix rows must have equal length".to_string(),
                    Position::NONE,
                )
                .into());
            }
            for (j, val) in row.iter().enumerate() {
                dm[(i, j)] = if val.is_float() {
                    val.as_float().unwrap()
                } else if val.is_int() {
                    val.as_int().unwrap() as FLOAT
                } else {
                    return Err(EvalAltResult::ErrorArithmetic(
                        "Matrix elements must be INT or FLOAT".to_string(),
                        Position::NONE,
                    )
                    .into());
                };
            }
        }
        Ok(dm)
    }

    /// Create a [`RhaiMatrix`] from a [`nalgebra::DMatrix`].
    #[cfg(feature = "nalgebra")]
    #[must_use]
    pub fn from_dmatrix(mat: &DMatrix<FLOAT>) -> Self {
        let mut rows = Vec::with_capacity(mat.nrows());
        for i in 0..mat.nrows() {
            let mut row = Vec::with_capacity(mat.ncols());
            for j in 0..mat.ncols() {
                row.push(Dynamic::from_float(mat[(i, j)]));
            }
            rows.push(Dynamic::from_array(row));
        }
        Self(rows)
    }

    /// Transpose the matrix.
    ///
    /// # Errors
    /// Returns an error if the matrix contains non-numeric values or rows of
    /// unequal length.
    #[cfg(feature = "nalgebra")]
    pub fn transpose(&self) -> Result<Self, Box<EvalAltResult>> {
        let dm = self.to_dmatrix()?;
        Ok(Self::from_dmatrix(&dm.transpose()))
    }

    /// Horizontally concatenate two matrices.
    ///
    /// # Errors
    /// Returns an error if the matrices have differing row counts or contain
    /// non-numeric values.
    #[cfg(feature = "nalgebra")]
    pub fn concat_h(&self, other: &Self) -> Result<Self, Box<EvalAltResult>> {
        let left = self.to_dmatrix()?;
        let right = other.to_dmatrix()?;
        if left.nrows() != right.nrows() {
            return Err(EvalAltResult::ErrorArithmetic(
                "Matrices must have the same number of rows".to_string(),
                Position::NONE,
            )
            .into());
        }
        let cols = left.ncols() + right.ncols();
        let rows = left.nrows();
        let mat = DMatrix::from_fn(rows, cols, |i, j| {
            if j < left.ncols() {
                left[(i, j)]
            } else {
                right[(i, j - left.ncols())]
            }
        });
        Ok(Self::from_dmatrix(&mat))
    }

    /// Vertically concatenate two matrices.
    ///
    /// # Errors
    /// Returns an error if the matrices have differing column counts or contain
    /// non-numeric values.
    #[cfg(feature = "nalgebra")]
    pub fn concat_v(&self, other: &Self) -> Result<Self, Box<EvalAltResult>> {
        let top = self.to_dmatrix()?;
        let bottom = other.to_dmatrix()?;
        if top.ncols() != bottom.ncols() {
            return Err(EvalAltResult::ErrorArithmetic(
                "Matrices must have the same number of columns".to_string(),
                Position::NONE,
            )
            .into());
        }
        let rows = top.nrows() + bottom.nrows();
        let cols = top.ncols();
        let mat = DMatrix::from_fn(rows, cols, |i, j| {
            if i < top.nrows() {
                top[(i, j)]
            } else {
                bottom[(i - top.nrows(), j)]
            }
        });
        Ok(Self::from_dmatrix(&mat))
    }
}

/// Wrapper around [`rhai::Array`] representing a vector.
///
/// # Examples
/// ```
/// use rhai::{Array, Dynamic};
/// use rhai_sci::matrix::RhaiVector;
/// let raw: Array = vec![Dynamic::from_float(1.0), Dynamic::from_float(2.0)];
/// let vector = RhaiVector::from_array(raw.clone());
/// assert_eq!(vector.to_array().len(), raw.len());
/// ```
#[derive(Clone, Debug)]
pub struct RhaiVector(Array);

impl RhaiVector {
    /// Construct a [`RhaiVector`] from a [`rhai::Array`].
    #[must_use]
    pub fn from_array(arr: Array) -> Self {
        Self(arr)
    }

    /// Convert the vector back into a [`rhai::Array`].
    #[must_use]
    pub fn to_array(self) -> Array {
        self.0
    }

    /// Convert the vector into a [`nalgebra::DVector`].
    ///
    /// # Errors
    /// Returns an error if any element is non-numeric.
    ///
    /// # Panics
    /// Panics if an integer value cannot be represented as `FLOAT`.
    #[cfg(feature = "nalgebra")]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_dvector(&self) -> Result<DVector<FLOAT>, Box<EvalAltResult>> {
        let mut dv = DVector::zeros(self.0.len());
        for (i, val) in self.0.iter().enumerate() {
            dv[i] = if val.is_float() {
                val.as_float().unwrap()
            } else if val.is_int() {
                val.as_int().unwrap() as FLOAT
            } else {
                return Err(EvalAltResult::ErrorArithmetic(
                    "Vector elements must be INT or FLOAT".to_string(),
                    Position::NONE,
                )
                .into());
            };
        }
        Ok(dv)
    }

    /// Create a [`RhaiVector`] from a [`nalgebra::DVector`].
    #[cfg(feature = "nalgebra")]
    #[must_use]
    pub fn from_dvector(vec: &DVector<FLOAT>) -> Self {
        let mut data = Vec::with_capacity(vec.len());
        for i in 0..vec.len() {
            data.push(Dynamic::from_float(vec[i]));
        }
        Self(data)
    }
}
