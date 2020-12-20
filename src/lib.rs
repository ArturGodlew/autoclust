mod edge;
use delaunator::{triangulate, Point};
use edge::{Edge, Graph, ToGraph};

pub fn autoclust(points: &[Point]) {
	let base_graph = triangulate(points).unwrap().to_graph(points);
	let phase1_graph = base_graph.filter_edges(&|g: &Graph, e: &Edge| {
		g.is_long(&g.verticies[e.vertex1], &g.verticies[e.vertex2])
			|| g.is_short(&g.verticies[e.vertex1], &g.verticies[e.vertex2])
	});
	let cc = phase1_graph.to_connected_components();
}

#[cfg(test)]
mod tests {
	#[test]
	fn it_works() {
		assert_eq!(2 + 2, 4);
	}
}
