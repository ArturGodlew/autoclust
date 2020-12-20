use adjacent_pair_iterator::AdjacentPairIterator;
use delaunator::{Point, Triangulation};
use itertools::Itertools;

#[derive(Clone, Copy, PartialEq)]
pub struct Edge {
	pub vertex1: usize,
	pub vertex2: usize,
}

#[derive(Clone)]
pub struct Graph {
	pub verticies: Vec<Vertex>,
	pub active_edges: Vec<Edge>,
	pub mean_std_deviation: f64,
}

pub struct ConnectedComponent {
	pub vertex_indices: Vec<usize>,
}

impl Graph {
	pub fn is_short(&self, from: &Vertex, to: &Vertex) -> bool {
		let length = distance(from, to);
		length < from.local_mean - self.mean_std_deviation
	}
	pub fn is_long(&self, from: &Vertex, to: &Vertex) -> bool {
		let length = distance(from, to);
		length > from.local_mean + self.mean_std_deviation
	}

	pub fn short_neighbours(&self, of: &Vertex) -> Vec<usize> {
		of.neighbours
			.iter()
			.filter(|i| {
				let neighbour = &self.verticies[**i];
				self.is_short(of, &neighbour)
			})
			.copied()
			.collect()
	}
	pub fn long_neighbours(&self, of: &Vertex) -> Vec<usize> {
		of.neighbours
			.iter()
			.filter(|i| {
				let neighbour = &self.verticies[**i];
				self.is_long(of, &neighbour)
			})
			.copied()
			.collect()
	}

	pub fn filter_edges(&self, predicate: &dyn Fn(&Graph, &Edge) -> bool) -> Graph {
		let mut result = self.clone();
		let edges_to_remove: Vec<Edge> = self
			.active_edges
			.iter()
			.filter(|e| predicate(self, e))
			.copied()
			.collect();
		result.active_edges.retain(|x| !edges_to_remove.contains(x));
		for edge in edges_to_remove {
			result.verticies[edge.vertex1]
				.neighbours
				.retain(|x| *x != edge.vertex2);
			result.verticies[edge.vertex2]
				.neighbours
				.retain(|x| *x != edge.vertex1);
		}
		result
	}

	fn build_connected_component(&self, cc: &mut Vec<usize>, vertex: &Vertex) {
		for neighbour in &vertex.neighbours {
			if !cc.contains(neighbour) {
				cc.push(*neighbour);
				self.build_connected_component(cc, &self.verticies[*neighbour]);
			}
		}
	}

	pub fn to_connected_components(&self) -> Vec<ConnectedComponent> {
		let mut result: Vec<ConnectedComponent> = vec![];
		while let Some(v) = self.verticies.iter().find(|x| {
			!matches!(
				result.iter().find(|y| y.vertex_indices.contains(&x.index)),
				None
			)
		}) {
			let mut cc = ConnectedComponent {
				vertex_indices: vec![],
			};
			self.build_connected_component(&mut cc.vertex_indices, v);
			result.push(cc);
		}
		result
	}
}

#[derive(Clone)]
pub struct Vertex {
	index: usize,
	point: Point,
	local_mean: f64,
	local_std_dev: f64,
	neighbours: Vec<usize>,
}

impl Vertex {
	fn new(index: usize, point: Point, neighbours: Vec<usize>) -> Vertex {
		Vertex {
			index,
			point,
			local_mean: 0.0,
			local_std_dev: 0.0,
			neighbours,
		}
	}
}

pub trait ToGraph {
	fn to_graph(&self, points: &[Point]) -> Graph;
}

impl ToGraph for Triangulation {
	fn to_graph(&self, points: &[Point]) -> Graph {
		let mut verticies: Vec<Vertex> = vec![];

		for i in 0..points.len() {
			let vertex = Vertex::new(
				i,
				Point {
					x: points[i].x,
					y: points[i].y,
				},
				neighborhood(i, self),
			);
			verticies.push(vertex);
		}
		for i in 0..points.len() {
			verticies[i].local_mean = local_mean(i, &verticies[i].neighbours, &verticies)
		}
		for i in 0..points.len() {
			verticies[i].local_std_dev =
				local_std_deviation(i, &verticies[i].neighbours, &verticies);
		}
		let global_mean_std_deviation =
			verticies.iter().fold(0.0, |acc, v| acc + v.local_mean) / (verticies.len() as f64);
		Graph {
			verticies,
			mean_std_deviation: global_mean_std_deviation,
			active_edges: self
				.halfedges
				.iter()
				.adjacent_pairs()
				.map(|(a, b)| Edge {
					vertex1: *a,
					vertex2: *b,
				})
				.collect(),
		}
	}
}

//impl EdgeDataSource for Point {
/* 	fn get_edge_data(&self, graph: &Triangulation, points: &[Point]) -> EdgeData {
	let index = points.iter().position(|x| x == self).unwrap();
	let neighbors = neighborhood(index, graph);
	EdgeData {
		index,
		local_mean: local_mean(index, neighbors, points),
		local_std_dev: local_std_deviation(index, neighbors, points),
		neighborhood: neighbors,
	}
} */
//}

fn distance(p1: &Vertex, p2: &Vertex) -> f64 {
	((p1.point.x - p2.point.x).powi(2) + (p1.point.y - p2.point.y).powi(2)).sqrt()
}

pub fn edges(point_index: usize, graph: &Triangulation) -> Vec<(usize, usize)> {
	graph
		.halfedges
		.iter()
		.batching(|it| match it.next() {
			None => None,
			Some(x) => match it.next() {
				None => None,
				Some(y) => Some((*x, *y)),
			},
		})
		.filter(|e| e.0 == point_index || e.1 == point_index)
		.collect()
}

fn neighborhood(point_index: usize, graph: &Triangulation) -> Vec<usize> {
	let edges = edges(point_index, graph);
	edges
		.iter()
		.map(|e| {
			if e.0 == point_index {
				return e.1;
			}
			e.0
		})
		.collect()
}

pub fn local_mean(point_index: usize, neighborhood: &[usize], points: &[Vertex]) -> f64 {
	let mut result = 0.0;
	for neighbour in neighborhood {
		result += distance(&points[point_index], &points[*neighbour]);
	}
	result
}
fn local_std_deviation(point_index: usize, neighborhood: &[usize], points: &[Vertex]) -> f64 {
	let mut result = 0.0;
	let cv = &points[point_index];
	for neighbour in neighborhood {
		let nv = &points[*neighbour];
		result += (nv.local_mean - distance(&cv, &nv)).powi(2) / (nv.neighbours.len() as f64);
	}
	result.sqrt()
}
