mod edge;
use delaunator::{triangulate, Point};
use edge::{Edge, FindLabel, Graph, ToGraph};

pub fn autoclust(points: &[Point]) {
	println!("pp{}", points.len());
	let base_graph = triangulate(points).unwrap().to_graph(points);
	let mut graph_with_filtered_edges = base_graph.filter_edges(&|g: &Graph, e: &Edge| {
		g.is_long(&g.verticies[e.vertex1], &g.verticies[e.vertex2])
			|| g.is_short(&g.verticies[e.vertex1], &g.verticies[e.vertex2])
	});
	let connected_components = graph_with_filtered_edges.to_connected_components();

	graph_with_filtered_edges.add_edges(&base_graph, &|v1, v2, g| {
		g.is_short(v1, v2)
			&& (!matches!(connected_components.find_label_for(&v1), None)
				^ !matches!(connected_components.find_label_for(&v2), None))
	});
}

#[cfg(test)]
mod tests {
	use delaunator::Point;
	#[test]
	fn it_works() {
		let points = [
			Point {
				x: 1.00007,
				y: 40.9378,
			},
			Point {
				x: 5.57577,
				y: 41.5869,
			},
			Point {
				x: 1.40539,
				y: 41.53,
			},
			Point {
				x: 3.55797,
				y: 41.8688,
			},
			Point {
				x: 2.68052,
				y: 41.3868,
			},
			Point {
				x: 2.95654,
				y: 41.486,
			},
			Point {
				x: -0.391807,
				y: 41.2361,
			},
			Point {
				x: 4.00381,
				y: 41.4221,
			},
			Point {
				x: 1.44796,
				y: 41.0426,
			},
		];
		crate::autoclust(&points);
	}
}
