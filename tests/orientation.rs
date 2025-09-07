use rhai::{Array, Dynamic};
use rhai_sci::matrix::RhaiMatrix;
use rhai_sci::validation_functions::{is_column_vector, is_row_vector};

#[test]
fn row_column_constructors_and_orientation() {
    let data: Array = vec![
        Dynamic::from_int(1),
        Dynamic::from_int(2),
        Dynamic::from_int(3),
    ];
    let row = RhaiMatrix::row_vector(data.clone());
    let column = RhaiMatrix::column_vector(data.clone());
    let mut row_to_col = row.as_column().unwrap().to_array();
    assert!(is_column_vector(&mut row_to_col));
    let mut col_to_row = column.as_row().unwrap().to_array();
    assert!(is_row_vector(&mut col_to_row));
}

#[test]
fn validate_orientation_helpers() {
    let mut row: Array = vec![Dynamic::from_array(vec![
        Dynamic::from_int(1),
        Dynamic::from_int(2),
    ])];
    let column: Array = vec![
        Dynamic::from_array(vec![Dynamic::from_int(1)]),
        Dynamic::from_array(vec![Dynamic::from_int(2)]),
    ];
    assert!(is_row_vector(&mut row.clone()));
    assert!(!is_row_vector(&mut column.clone()));
    assert!(is_column_vector(&mut column.clone()));
    assert!(!is_column_vector(&mut row));
}
