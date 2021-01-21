mod edge;
use delaunator::{triangulate, Point};
use edge::{ToGraph, Vertex};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn autoclust_implementation(points: &[Point]) -> (Vec<Vertex>, Vec<Vertex>, Vec<Vertex>) {
	let base_graph = triangulate(points).unwrap().to_graph(points);
	let mut graph_with_filtered_edges = base_graph.filter_edges();
	graph_with_filtered_edges.calculate_connected_components();

	let v1 = graph_with_filtered_edges.verticies.clone();

	graph_with_filtered_edges.restore_edges();

	let v2 = graph_with_filtered_edges.verticies.clone();

	graph_with_filtered_edges.recalculate_mean_with_k_neighbourhood();

	graph_with_filtered_edges.calculate_connected_components();

	(v1, v2, graph_with_filtered_edges.verticies)
}

#[pyfunction]
pub fn auto_clust(input: Vec<P>) -> PyResult<(Vec<Res>, Vec<Res>, Vec<Res>)> {
	let points: Vec<Point> = input.iter().map(|x| Point { x: x.x, y: x.y }).collect();
	let (v1, v2, v3) = autoclust_implementation(&points);
	let mut result1: Vec<Res> = vec![];
	let mut result2: Vec<Res> = vec![];
	let mut result3: Vec<Res> = vec![];
	for v in v1.iter() {
		result1.push(Res {
			x: v.point.x,
			y: v.point.y,
			label: v.label,
		})
	}

	for v in v2.iter() {
		result2.push(Res {
			x: v.point.x,
			y: v.point.y,
			label: v.label,
		})
	}

	for v in v3.iter() {
		result3.push(Res {
			x: v.point.x,
			y: v.point.y,
			label: v.label,
		})
	}
	Ok((result1, result2, result3))
}

#[pyclass]
#[derive(FromPyObject)]
pub struct P {
	#[pyo3(get, set)]
	pub x: f64,
	#[pyo3(get, set)]
	pub y: f64,
}

#[pymethods]
impl P {
	#[new]
	pub fn new(x: f64, y: f64) -> Self {
		P { x, y }
	}
}

#[pyclass]
#[derive(Debug)]
pub struct Res {
	#[pyo3(get, set)]
	pub x: f64,
	#[pyo3(get, set)]
	pub y: f64,
	#[pyo3(get, set)]
	pub label: usize,
}

#[pymodule]
pub fn autoclust(_: Python, m: &PyModule) -> PyResult<()> {
	m.add_class::<P>()?;
	m.add_class::<Res>()?;
	m.add_wrapped(wrap_pyfunction!(auto_clust))?;
	Ok(())
}
