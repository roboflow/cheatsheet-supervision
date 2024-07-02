export default [
    `\
vertex_annotator = sv.VertexAnnotator(radius=10)
edge_annotator = sv.EdgeAnnotator(thickness=5)

annotated_frame = edge_annotator.annotate(
    scene=image.copy(),
    key_points=key_points
)
annotated_frame = vertex_annotator.annotate(
    scene=annotated_frame,
    key_points=key_points
)`
];