use rhai::{Array, Dynamic, FLOAT, INT};
use rhai_sci::matrix::RhaiMatrix;
use rhai_sci::matrix_functions::{horzcat, matrix_size_by_reference, repmat, transpose, vertcat};
use rhai_sci::validation_functions::{is_column_vector, is_row_vector};

#[test]
fn transpose_orients_row_vector() {
    let data: Array = vec![
        Dynamic::from_int(1),
        Dynamic::from_int(2),
        Dynamic::from_int(3),
    ];
    let row = RhaiMatrix::row_vector(data);
    let mut result = transpose(row).unwrap().to_array();
    assert!(is_column_vector(&mut result));
}

#[test]
fn horzcat_concatenates_rows() {
    let a: Array = vec![Dynamic::from_int(1), Dynamic::from_int(2)];
    let b: Array = vec![Dynamic::from_int(3), Dynamic::from_int(4)];
    let m1 = RhaiMatrix::row_vector(a);
    let m2 = RhaiMatrix::row_vector(b);
    let mut result = horzcat(m1, m2).unwrap().to_array();
    assert!(is_row_vector(&mut result));
    let row = result[0].clone().into_array().unwrap();
    let values: Vec<FLOAT> = row.into_iter().map(|d| d.as_float().unwrap()).collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn vertcat_concatenates_columns() {
    let a: Array = vec![
        Dynamic::from_array(vec![Dynamic::from_int(1)]),
        Dynamic::from_array(vec![Dynamic::from_int(2)]),
    ];
    let b: Array = vec![
        Dynamic::from_array(vec![Dynamic::from_int(3)]),
        Dynamic::from_array(vec![Dynamic::from_int(4)]),
    ];
    let m1 = RhaiMatrix::from_array(a);
    let m2 = RhaiMatrix::from_array(b);
    let mut result = vertcat(m1, m2).unwrap().to_array();
    assert!(is_column_vector(&mut result));
}

#[test]
fn repmat_replicates_matrix() {
    let data: Array = vec![
        Dynamic::from_array(vec![Dynamic::from_int(1), Dynamic::from_int(2)]),
        Dynamic::from_array(vec![Dynamic::from_int(3), Dynamic::from_int(4)]),
    ];
    let m = RhaiMatrix::from_array(data);
    let mut result = repmat(m, 2, 2).unwrap().to_array();
    let shape = matrix_size_by_reference(&mut result);
    let dims: Vec<INT> = shape.into_iter().map(|d| d.as_int().unwrap()).collect();
    assert_eq!(dims, vec![4, 4]);
}
