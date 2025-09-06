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
