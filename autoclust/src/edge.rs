use delaunator::{Point, Triangulation};
use itertools::Itertools;

#[derive(Clone, Copy, PartialEq)]
pub struct Edge {
	pub vertex1: usize,
	pub vertex2: usize,
}

impl Edge {
	pub fn other(&self, index: usize) -> usize {
		if self.vertex1 == index {
			return self.vertex2;
		}
		self.vertex1
	}
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

pub trait FindLabel {
	fn find_label_for(&self, vertex: &Vertex) -> Option<usize>;
	fn find_label_for_index(&self, index: usize) -> Option<usize>;
}

pub trait Reasign {
	fn reasign(&mut self, from: usize, to: usize);
}

impl Reasign for Vec<ConnectedComponent> {
	fn reasign(&mut self, what: usize, to: usize) {
		{
			if let Some(from_cc) = self.iter_mut().find(|x| x.vertex_indices.contains(&what)) {
				let index = from_cc
					.vertex_indices
					.iter()
					.position(|x| *x == what)
					.unwrap();
				from_cc.vertex_indices.remove(index);
			}
		}
		self[to - 1].vertex_indices.push(what);
	}
}

impl FindLabel for [ConnectedComponent] {
	fn find_label_for(&self, vertex: &Vertex) -> Option<usize> {
		self.iter()
			.position(|cc| cc.vertex_indices.contains(&vertex.index))
	}
	fn find_label_for_index(&self, index: usize) -> Option<usize> {
		match self
			.iter()
			.position(|cc| cc.vertex_indices.contains(&index))
		{
			None => None,
			Some(x) => Some(x + 1),
		}
	}
}

impl Graph {
	fn edge_is_active(&self, v1: usize, v2: usize) -> bool {
		matches!(
			self.active_edges
				.iter()
				.find(|x| (x.vertex1 == v1 && x.vertex2 == v2)
					|| (x.vertex2 == v1 && x.vertex1 == v2)),
			Some(_)
		)
	}

	pub fn is_short(&self, from: &Vertex, to: &Vertex) -> bool {
		let length = distance(from, to);
		length < from.local_mean - self.mean_std_deviation
	}
	pub fn is_long(&self, from: &Vertex, to: &Vertex) -> bool {
		let length = distance(from, to);
		length > from.local_mean + self.mean_std_deviation
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
		cc.push(vertex.index);
		for neighbour in vertex
			.neighbours
			.iter()
			.filter(|x| self.edge_is_active(**x, vertex.index))
		{
			if !cc.contains(neighbour) {
				cc.push(*neighbour);
				self.build_connected_component(cc, &self.verticies[*neighbour]);
			}
		}
	}

	pub fn to_connected_components(&self) -> Vec<ConnectedComponent> {
		let mut result: Vec<ConnectedComponent> = vec![];

		while let Some(v) = self.verticies.iter().find(|x| {
			matches!(
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
		result.retain(|x| x.vertex_indices.len() > 1);
		result
	}

	pub fn add_edges(
		&mut self,
		base: &Graph,
		predicate: &dyn Fn(&Vertex, &Vertex, &Graph) -> bool,
	) {
		for edge in &base.active_edges {
			if !self.active_edges.contains(&edge)
				&& predicate(
					&self.verticies[edge.vertex1],
					&self.verticies[edge.vertex2],
					&self,
				) {
				self.active_edges.push(*edge);
				self.verticies[edge.vertex1].neighbours.push(edge.vertex2);
				self.verticies[edge.vertex2].neighbours.push(edge.vertex1);
			}
		}
	}

	pub fn restore_edges(&mut self, base_graph: &Graph, cc: &mut Vec<ConnectedComponent>) {
		struct LabelReference {
			size: usize,
			label: usize,
			largest_index: usize,
		};
		for i in 0..self.verticies.len() {
			let short_edges: Vec<&Edge> = base_graph
				.find_edges(i)
				.iter()
				.filter(|e| self.is_short(&self.verticies[e.vertex1], &self.verticies[e.vertex2]))
				.copied()
				.collect();
			let label = cc.find_label_for_index(i);
			let mut possible_labels: Vec<LabelReference> = vec![];
			for e in short_edges {
				if let Some(other_label) = cc.find_label_for_index(e.other(i)) {
					if label != Some(other_label) {
						let other_size = cc[other_label - 1].vertex_indices.len();
						if let Some(existing_reference) =
							possible_labels.iter_mut().find(|x| x.label == other_label)
						{
							if existing_reference.size > other_size {
								existing_reference.size = other_size
							}
						} else {
							possible_labels.push(LabelReference {
								size: other_size,
								label: other_label,
								largest_index: other_label - 1,
							})
						}
					}
				}
			}
			if let Some(best_label) = possible_labels.iter().max_by_key(|x| x.size) {
				let current_label_size = match label {
					Some(x) => cc[x - 1].vertex_indices.len(),
					None => 0,
				};
				if Some(best_label.label) != label && best_label.size > current_label_size {
					cc.reasign(i, best_label.label);
					self.active_edges.push(Edge {
						vertex1: i,
						vertex2: best_label.largest_index,
					});
				}
			}
		}
	}

	fn find_edges(&self, vertex_index: usize) -> Vec<&Edge> {
		self.active_edges
			.iter()
			.filter(|e| e.vertex1 == vertex_index || e.vertex2 == vertex_index)
			.collect()
	}
}

#[derive(Clone, Debug)]
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
			let ne = neighborhood(i, self);
			for n in &ne {
				assert!(n < &points.len());
			}
			let vertex = Vertex::new(
				i,
				Point {
					x: points[i].x,
					y: points[i].y,
				},
				ne,
			);
			verticies.push(vertex);
		}
		for i in 0..points.len() {
			verticies[i].local_mean = local_mean(i, &verticies[i].neighbours.as_slice(), &verticies)
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
			active_edges: all_edges(self)
				.iter()
				.map(|e| Edge {
					vertex1: self.triangles[e.0],
					vertex2: self.triangles[e.1],
				})
				.collect(),
		}
	}
}

fn distance(p1: &Vertex, p2: &Vertex) -> f64 {
	((p1.point.x - p2.point.x).powi(2) + (p1.point.y - p2.point.y).powi(2)).sqrt()
}

pub fn all_edges(graph: &Triangulation) -> Vec<(usize, usize)> {
	let mut result: Vec<(usize, usize)> = vec![];
	for t in graph.triangles.iter().batching(|it| match it.next() {
		None => None,
		Some(x) => match it.next() {
			None => None,
			Some(y) => match it.next() {
				None => None,
				Some(z) => Some((*x, *y, *z)),
			},
		},
	}) {
		result.push((t.0, t.1));
		result.push((t.1, t.2));
		result.push((t.2, t.0));
	}
	result
}

pub fn edges(point_index: usize, graph: &Triangulation) -> Vec<(usize, usize)> {
	let result = all_edges(graph);
	result
		.iter()
		.filter(|x| x.0 == point_index || x.1 == point_index)
		.copied()
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
	result / (neighborhood.len() as f64)
}
fn local_std_deviation(point_index: usize, neighborhood: &[usize], points: &[Vertex]) -> f64 {
	let mut result = 0.0;
	let cv = &points[point_index];
	for neighbour in neighborhood {
		let nv = &points[*neighbour];
		result += (cv.local_mean - distance(&cv, &nv)).powi(2) / (cv.neighbours.len() as f64);
	}
	result.sqrt()
}

#[test]
fn name() {
	let verticies = vec![
		Vertex {
			index: 0,
			point: Point { x: 0.0, y: 0.0 },
			local_mean: 0.0,
			local_std_dev: 0.0,
			neighbours: vec![],
		},
		Vertex {
			index: 0,
			point: Point { x: 0.0, y: 0.0 },
			local_mean: 0.0,
			local_std_dev: 0.0,
			neighbours: vec![],
		},
		Vertex {
			index: 0,
			point: Point { x: 0.0, y: 0.0 },
			local_mean: 0.0,
			local_std_dev: 0.0,
			neighbours: vec![],
		},
		Vertex {
			index: 0,
			point: Point { x: 0.0, y: 0.0 },
			local_mean: 0.0,
			local_std_dev: 0.0,
			neighbours: vec![],
		},
	];
	let local_mean = local_mean(0, &[1, 2], &verticies);
}
