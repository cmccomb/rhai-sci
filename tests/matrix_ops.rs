use rhai::{Array, Dynamic};
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
    assert_eq!(
        result,
        vec![Dynamic::from_array(vec![
            Dynamic::from_int(1),
            Dynamic::from_int(2),
            Dynamic::from_int(3),
            Dynamic::from_int(4),
        ])]
    );
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
    assert_eq!(shape, vec![Dynamic::from_int(4), Dynamic::from_int(4)]);
}
