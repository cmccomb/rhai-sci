use rhai::{packages::Package, Array, Engine, EvalAltResult};
use rhai_sci::SciPackage;

#[test]
fn constructors_make_vector_orientation_explicit() {
    assert_matrix_eq(eval_array("row([1, 2, 3])").unwrap(), &[&[1.0, 2.0, 3.0]]);
    assert_matrix_eq(
        eval_array("col([1, 2, 3])").unwrap(),
        &[&[1.0], &[2.0], &[3.0]],
    );
    assert_matrix_eq(
        eval_array("vec([1, 2, 3])").unwrap(),
        &[&[1.0], &[2.0], &[3.0]],
    );
}

#[test]
fn string_literals_accept_spaces_commas_and_semicolons() {
    assert_matrix_eq(
        eval_array("mat(\"1, 2; 3, 4\")").unwrap(),
        &[&[1.0, 2.0], &[3.0, 4.0]],
    );
    assert_matrix_eq(
        eval_array("M(\"1 2; 3 4\")").unwrap(),
        &[&[1.0, 2.0], &[3.0, 4.0]],
    );
    assert_matrix_eq(eval_array("R(\"1 2 3\")").unwrap(), &[&[1.0, 2.0, 3.0]]);
    assert_matrix_eq(
        eval_array("C(\"1; 2; 3\")").unwrap(),
        &[&[1.0], &[2.0], &[3.0]],
    );
}

#[test]
fn aliases_read_like_linear_algebra() {
    let product = eval_array(
        r#"
            let A = mat("1 2; 3 4");
            let x = col([5, 6]);
            dot(A, x)
        "#,
    )
    .unwrap();
    assert_matrix_eq(product, &[&[17.0], &[39.0]]);

    let method_product = eval_array(
        r#"
            let A = mat("1 2; 3 4");
            let x = col([5, 6]);
            A.dot(x)
        "#,
    )
    .unwrap();
    assert_matrix_eq(method_product, &[&[17.0], &[39.0]]);

    assert_matrix_eq(eval_array("T(row([1, 2]))").unwrap(), &[&[1.0], &[2.0]]);
    assert_matrix_eq(
        eval_array("hcat(mat(\"1 2; 3 4\"), col([5, 6]))").unwrap(),
        &[&[1.0, 2.0, 5.0], &[3.0, 4.0, 6.0]],
    );
    assert_matrix_eq(
        eval_array("vcat(mat(\"1 2; 3 4\"), row([5, 6]))").unwrap(),
        &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]],
    );
}

#[test]
fn matrix_constructor_rejects_ragged_literals() {
    assert_error_contains("mat(\"1 2; 3\")", "equal length");
}

#[test]
fn matrix_constructor_rejects_invalid_arrays() {
    assert_error_contains("mat([[1, 2], [3]])", "equal length");
    assert_error_contains("mat([[1, \"x\"]])", "INT or FLOAT");
    assert_error_contains("mat([[]])", "at least one value");
}

#[test]
fn vector_constructors_reject_empty_arrays() {
    assert_error_contains("vec([])", "at least one value");
    assert_error_contains("row([])", "at least one value");
    assert_error_contains("col([])", "at least one value");
}

fn eval_array(script: &str) -> Result<Array, Box<EvalAltResult>> {
    let mut engine = Engine::new();
    engine.register_global_module(SciPackage::new().as_shared_module());
    engine.eval::<Array>(script)
}

fn assert_error_contains(script: &str, expected: &str) {
    let err = eval_array(script).unwrap_err();
    match err.as_ref() {
        EvalAltResult::ErrorArithmetic(message, _) => {
            assert!(
                message.contains(expected),
                "expected error message `{message}` to contain `{expected}`"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

fn assert_matrix_eq(actual: Array, expected: &[&[f64]]) {
    let actual = numeric_matrix(actual);
    let expected = expected
        .iter()
        .map(|row| row.to_vec())
        .collect::<Vec<Vec<f64>>>();
    assert_eq!(actual, expected);
}

fn numeric_matrix(matrix: Array) -> Vec<Vec<f64>> {
    matrix
        .into_iter()
        .map(|row| {
            row.into_array()
                .expect("matrix rows should be arrays")
                .into_iter()
                .map(|value| {
                    if value.is_float() {
                        value.as_float().expect("value should be FLOAT")
                    } else {
                        value.as_int().expect("value should be INT") as f64
                    }
                })
                .collect()
        })
        .collect()
}
