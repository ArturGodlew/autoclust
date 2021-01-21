use delaunator::{Point, Triangulation};
use itertools::Itertools;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Edge {
	pub vertex1: usize,
	pub vertex2: usize,
	pub edge_type1: EdgeType,
	pub edge_type2: EdgeType,
	pub active: bool,
	pub length: f64,
}

#[derive(PartialEq, Copy, Clone)]
pub enum EdgeType {
	Long,
	Short,
	Medium,
	Empty,
}

impl Edge {
	pub fn other(&self, index: usize) -> usize {
		if self.vertex1 == index {
			return self.vertex2;
		}
		self.vertex1
	}

	pub fn edge_type(&self, perspective: usize) -> EdgeType {
		if self.vertex1 == perspective {
			return self.edge_type1;
		}
		self.edge_type2
	}

	pub fn new(v1: usize, v2: usize, t1: EdgeType, t2: EdgeType, length: f64) -> Edge {
		Edge {
			vertex1: v1,
			vertex2: v2,
			edge_type1: t1,
			edge_type2: t2,
			active: true,
			length,
		}
	}
}

impl PartialEq for Edge {
	fn eq(&self, other: &Self) -> bool {
		(self.vertex1 == other.vertex1 && self.vertex2 == other.vertex2)
			|| (self.vertex1 == other.vertex2 && self.vertex2 == other.vertex1)
	}
}

#[derive(Clone)]
pub struct Graph {
	pub verticies: Vec<Vertex>,
	pub active_edges: Vec<Edge>,
	pub mean_std_deviation: f64,
}

pub trait FindLabel {
	fn find_label_for(&self, vertex: &Vertex) -> Option<usize>;
	fn find_label_for_index(&self, index: usize) -> Option<usize>;
}

pub trait Reasign {
	fn reasign(&mut self, from: usize, to: usize);
}

impl Graph {
	fn edge_is_active(&self, e: usize) -> bool {
		self.active_edges[e].active
	}

	pub fn is_short(&self, from: &Vertex, to: &Vertex) -> bool {
		let length = distance(from, to);
		length < from.local_mean - self.mean_std_deviation
	}
	pub fn is_long(&self, from: &Vertex, to: &Vertex) -> bool {
		let length = distance(from, to);
		length > from.local_mean + self.mean_std_deviation
	}

	pub fn calculate_type(&self, from: &Vertex, to: &Vertex) -> EdgeType {
		if self.is_long(from, to) {
			return EdgeType::Long;
		}
		if self.is_short(from, to) {
			return EdgeType::Short;
		}
		EdgeType::Medium
	}

	pub fn filter_edges(&self) -> Graph {
		let mut result = self.clone();
		for edge in result.active_edges.iter_mut() {
			if edge.edge_type1 != EdgeType::Medium && edge.edge_type2 != EdgeType::Medium {
				edge.active = false;
			}
		}
		result
	}

	fn build_connected_component(&mut self, vertex_index: usize, label: usize) {
		if self.verticies[vertex_index].label != label {
			self.verticies[vertex_index].label = label;
			for i in 0..self.verticies[vertex_index].edges.len() {
				let edge_index = self.verticies[vertex_index].edges[i];
				if self.edge_is_active(edge_index)
					&& self.verticies[self.active_edges[edge_index].other(vertex_index)].label == 0
				{
					self.build_connected_component(
						self.active_edges[edge_index].other(vertex_index),
						label,
					);
				}
			}
		}
	}

	pub fn calculate_connected_components(&mut self) {
		let mut cc_index = 1;
		while let Some(v) = self
			.verticies
			.iter_mut()
			.position(|x| !x.edges.is_empty() && x.label == 0)
		{
			self.build_connected_component(v, cc_index);
			cc_index += 1;
		}

		let groups = self.calculate_cc_sizes();
		for (label, size) in groups {
			if size == 1 {
				for v in 0..self.verticies.len() {
					if self.verticies[v].label == label {
						self.verticies[v].label = 0;
						break;
					}
				}
			}
		}
	}

	fn calculate_cc_sizes(&self) -> HashMap<usize, usize> {
		let mut cc_sizes: HashMap<usize, usize> = HashMap::new();
		for vertex in &self.verticies {
			*cc_sizes.entry(vertex.label).or_insert(0) += 1;
		}
		cc_sizes
	}

	fn reassign(
		&mut self,
		vertex_index: usize,
		label: usize,
		cc_sizes: &mut HashMap<usize, usize>,
	) {
		if self.verticies[vertex_index].label != label {
			*cc_sizes
				.get_mut(&self.verticies[vertex_index].label)
				.unwrap() -= 1;
			*cc_sizes.get_mut(&label).unwrap() += 1;
			let vertex = &mut self.verticies[vertex_index];
			vertex.label = label;
			for e in 0..vertex.edges.len() {
				let edge = self.verticies[vertex_index].edges[e];
				let other = self.active_edges[edge].other(vertex_index);
				if self.active_edges[edge].edge_type(vertex_index) == EdgeType::Short
					&& self.verticies[other].label == label
				{
					self.active_edges[edge].active = true;
				}
				if self.verticies[other].label != label {
					self.active_edges[edge].active = false;
				}
			}
		}
	}

	pub fn restore_edges(&mut self) {
		struct LabelReference {
			size: usize,
			label: usize,
			edge_index: usize,
		};
		let mut cc_sizes = self.calculate_cc_sizes();
		let mut reassign_map: HashMap<usize, usize> = HashMap::new();
		for i in 0..self.verticies.len() {
			let short_edges: Vec<&Edge> = self.verticies[i]
				.edges
				.iter()
				.filter(|e| self.active_edges[**e].edge_type(i) == EdgeType::Short)
				.map(|x| &self.active_edges[*x])
				.collect();
			let label = self.verticies[i].label;
			let mut possible_labels: Vec<LabelReference> = vec![];
			for (i, e) in short_edges.iter().enumerate() {
				let other_label = self.verticies[e.other(i)].label;
				if other_label != 0 && label != other_label {
					let other_size = cc_sizes[&other_label];
					if matches!(
						possible_labels.iter_mut().find(|x| x.label == other_label),
						None
					) {
						possible_labels.push(LabelReference {
							size: other_size,
							label: other_label,
							edge_index: i,
						})
					}
				}
			}
			if let Some(best_label) = possible_labels.iter().max_by_key(|x| x.size) {
				if best_label.label != label {
					*reassign_map.entry(i).or_insert(0) = best_label.label;
				}
			}
		}

		for (vertex, label) in reassign_map {
			self.reassign(vertex, label, &mut cc_sizes);
		}

		for i in 0..self.verticies.len() {
			for &edge in self.verticies[i].edges.iter() {
				if self.active_edges[edge].edge_type(i) == EdgeType::Short
					&& self.verticies[self.active_edges[edge].other(i)].label
						== self.verticies[i].label
				{
					self.active_edges[edge].active = true;
				}
			}
		}
	}

	fn recalculate_k_neighbourhood(&mut self, vertex_index: usize) {
		let mut edge_count: usize = 0;
		let mut edge_sum = 0.0;
		for &edge1 in self.verticies[vertex_index].edges.iter() {
			if self.edge_is_active(edge1) {
				let other = self.active_edges[edge1].other(vertex_index);

				for &edge2 in self.verticies[other].edges.iter() {
					if self.edge_is_active(edge2) {
						edge_sum += self.active_edges[edge2].length;
						edge_count += 1;
					}
				}
			}
		}
		self.verticies[vertex_index].local_mean = edge_sum / edge_count as f64;
		self.verticies[vertex_index].local_std_dev =
			local_std_deviation(vertex_index, &self.active_edges, &self.verticies)
	}

	pub fn recalculate_mean_with_k_neighbourhood(&mut self) {
		for v in 0..self.verticies.len() {
			self.recalculate_k_neighbourhood(v);
			self.verticies[v].label = 0;
		}

		self.mean_std_deviation = self.mean_std_deviation();
		for v in 0..self.verticies.len() {
			for e in 0..self.verticies[v].edges.len() {
				let other = self.active_edges[self.verticies[v].edges[e]].other(v);
				for e2 in 0..self.verticies[other].edges.len() {
					let is_long = self.active_edges[self.verticies[other].edges[e2]].length
						> self.verticies[v].local_mean + self.mean_std_deviation;
					if is_long {
						self.active_edges[self.verticies[other].edges[e2]].active = false;
						self.mean_std_deviation = self.mean_std_deviation();
						self.verticies[v].local_std_dev =
							local_std_deviation(v, &self.active_edges, &self.verticies);
					}
				}
			}
		}
	}

	fn mean_std_deviation(&self) -> f64 {
		self.verticies
			.iter()
			.fold(0.0, |acc, v| acc + v.local_std_dev)
			/ self.verticies.len() as f64
	}
}

#[derive(Clone, Debug)]
pub struct Vertex {
	index: usize,
	pub point: Point,
	local_mean: f64,
	local_std_dev: f64,
	edges: Vec<usize>,
	pub label: usize,
}

impl Vertex {
	fn new(index: usize, point: Point, edges: Vec<usize>) -> Vertex {
		Vertex {
			index,
			point,
			local_mean: 0.0,
			local_std_dev: 0.0,
			edges,
			label: 0,
		}
	}
}

pub trait ToGraph {
	fn to_graph(&self, points: &[Point]) -> Graph;
}

impl ToGraph for Triangulation {
	fn to_graph(&self, points: &[Point]) -> Graph {
		let all_edges = all_edges(self, points);
		let mut verticies: Vec<Vertex> = vec![];

		for (i, p) in points.iter().enumerate() {
			let vertex = Vertex::new(i, Point { x: p.x, y: p.y }, neighborhood(i, &all_edges));
			verticies.push(vertex);
		}
		for v in verticies.iter_mut() {
			v.local_mean = local_mean(&all_edges, &v.edges)
		}
		for i in 0..points.len() {
			verticies[i].local_std_dev = local_std_deviation(i, &all_edges, &verticies);
		}
		let mut result = Graph {
			verticies,
			mean_std_deviation: 0.0,
			active_edges: all_edges,
		};
		result.mean_std_deviation = result.mean_std_deviation();
		for i in 0..result.active_edges.len() {
			result.active_edges[i].edge_type1 = result.calculate_type(
				&result.verticies[result.active_edges[i].vertex1],
				&result.verticies[result.active_edges[i].vertex2],
			);
			result.active_edges[i].edge_type2 = result.calculate_type(
				&result.verticies[result.active_edges[i].vertex2],
				&result.verticies[result.active_edges[i].vertex1],
			);
		}
		result
	}
}

fn distance(p1: &Vertex, p2: &Vertex) -> f64 {
	((p1.point.x - p2.point.x).powi(2) + (p1.point.y - p2.point.y).powi(2)).sqrt()
}

fn distance_point(p1: &Point, p2: &Point) -> f64 {
	((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt()
}

pub fn all_edges(graph: &Triangulation, points: &[Point]) -> Vec<Edge> {
	let mut result: Vec<Edge> = vec![];
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
		let e1 = Edge::new(
			t.0,
			t.1,
			EdgeType::Empty,
			EdgeType::Empty,
			distance_point(&points[t.0], &points[t.1]),
		);
		let e2 = Edge::new(
			t.1,
			t.2,
			EdgeType::Empty,
			EdgeType::Empty,
			distance_point(&points[t.1], &points[t.2]),
		);
		let e3 = Edge::new(
			t.2,
			t.0,
			EdgeType::Empty,
			EdgeType::Empty,
			distance_point(&points[t.2], &points[t.0]),
		);
		if !result.contains(&e1) {
			result.push(e1);
		}
		if !result.contains(&e2) {
			result.push(e2);
		}
		if !result.contains(&e3) {
			result.push(e3);
		}
	}
	result
}

pub fn edges(point_index: usize, all_edges: &[Edge]) -> Vec<usize> {
	let mut result: Vec<usize> = vec![];
	for (i, edge) in all_edges.iter().enumerate() {
		if edge.vertex1 == point_index || edge.vertex2 == point_index {
			result.push(i);
		}
	}
	result
}

fn neighborhood(point_index: usize, all_edges: &[Edge]) -> Vec<usize> {
	edges(point_index, all_edges)
}

pub fn local_mean(edges: &[Edge], edge_indicies: &[usize]) -> f64 {
	let mut result = 0.0;
	for index in edge_indicies {
		result += edges[*index].length;
	}
	result / (edge_indicies.len() as f64)
}
fn local_std_deviation(point_index: usize, edges: &[Edge], points: &[Vertex]) -> f64 {
	let mut result = 0.0;
	let current_vertex = &points[point_index];
	for edge in current_vertex
		.edges
		.iter()
		.map(|x| &edges[*x])
		.filter(|x| x.active)
	{
		result += (current_vertex.local_mean - edge.length).powi(2)
			/ (current_vertex
				.edges
				.iter()
				.map(|x| &edges[*x])
				.filter(|x| x.active)
				.count() as f64);
	}
	result.sqrt()
}
