"""Load human-readable golden JSON fixtures into IR types."""

from pathlib import Path

from lucid.ir.perception import PerceptualEvidenceGraph
from lucid.ir.serde import from_json

FIXTURES = Path(__file__).parent / "fixtures"


def test_perception_graph_fixture_roundtrip():
    path = FIXTURES / "perception_graph.json"
    text = path.read_text(encoding="utf-8")
    graph = from_json(text, PerceptualEvidenceGraph)
    assert graph.candidate_units[0].surface == "bank"
    assert graph.provenance.modality.value == "text"
