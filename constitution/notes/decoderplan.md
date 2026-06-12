Here’s a clean document-style version you can use as the base spec.
:::writing{variant="document" id="48291"}
Lucid Decoder Architecture
1. Purpose of the Decoder
The decoder is the final language layer in the Lucid system.
Lucid is responsible for understanding, reasoning, validation, and deciding what is safe to express. The decoder does not do any of those jobs again. Its only purpose is to take Lucid’s approved render packet and turn it into polished user-facing language.
In simple terms:
Plain text
Lucid decides what can be said.
The decoder decides how to say it.
The decoder should not invent facts, add new reasoning, introduce new entities, or expand the answer using outside knowledge. It should only express the approved meaning in fluent language.
The decoder exists because Lucid’s output is structured and machine-friendly, while the user needs an answer that is clear, natural, and easy to read.
2. Core Principle
The decoder follows one main rule:
Plain text
Approved meaning in → fluent language out
It should be flexible in wording but strict in meaning.
This means the decoder can choose natural phrasing, sentence flow, and wording, but it cannot change the approved meaning. It is not allowed to “make the answer better” by adding information that Lucid did not approve.
For example, if Lucid approves:
Plain text
qubit type_of unit of quantum information
qubit property superposition possible
The decoder may say:
Plain text
A qubit is a unit of quantum information that can be in a superposition state.
But it should not say:
Plain text
A qubit is like a classical bit that can be both 0 and 1 at the same time.
Even though that may sound helpful, it introduces new entities and ideas that were not present in the approved packet: “classical bit,” “0,” “1,” and a comparison between classical and quantum bits.
The decoder must stay inside the approved meaning boundary.
3. What the Decoder Is Not
The decoder is not a second reasoning engine.
It should not:
Plain text
re-answer the user’s question
reinterpret the original prompt
add new facts
add examples unless approved
add analogies unless approved
add causal links unless approved
decide whether the answer is safe
choose between competing reasoning paths
Those responsibilities belong to Lucid.
The decoder should also not be a handcoded template system. A template system would produce rigid answers and would not generalize well. The decoder should learn how to express meaning from text, but its output must still be checked against Lucid’s approved packet.
4. Minimal Decoder Architecture
The decoder is intentionally small.
Plain text
Lucid Render Packet
        ↓
1. Canvas Builder
        ↓
2. Denoising Realizer
        ↓
3. Faithfulness Check
        ↓
Final Answer
This is the debloated architecture.
There is no separate meaning frame builder, no discourse planner, no packet permission gate, and no candidate selector. Lucid already handles approval and commitment, so the decoder should not duplicate that work.
The decoder only needs three layers:
Plain text
Canvas Builder
Denoising Realizer
Faithfulness Check
Each layer has a clear job.
5. Layer 1: Canvas Builder
Purpose
The Canvas Builder converts Lucid’s approved render packet into a rough text canvas.
The canvas is not the final answer. It is a simple intermediate form that makes the approved graph easier for the language layer to express.
The Canvas Builder does not reason.
It does not invent.
It does not polish.
It only turns approved graph edges into rough sayable text.
Input
The input is Lucid’s approved render packet.
Example:
Plain text
concept:qubit type_of concept:unit_of_quantum_information
concept:qubit property property:superposition_possible
Output
The Canvas Builder converts that into rough text:
Plain text
qubit is unit of quantum information
qubit can be in superposition
This rough canvas is allowed to be ugly. Its main job is to preserve the approved facts.
Example: Physical Reasoning
Lucid-approved edges:
Plain text
man inside_or_on car
car moves away
man at away
Canvas:
Plain text
man is inside or on car
car moves away
man ends up away
The canvas is still not polished language, but it gives the next layer a clean starting point.
Why This Layer Exists
The Denoising Realizer should not have to read the full Lucid object directly. The render packet may contain policy fields, receipts, graph metadata, source refs, confidence values, and audit notes.
The Canvas Builder strips away the complexity and gives the realizer a direct expression surface:
Plain text
approved facts → rough language
This makes the decoder easier to train and easier to debug.
6. Layer 2: Denoising Realizer
Purpose
The Denoising Realizer is the language-producing layer.
It takes the rough canvas and turns it into polished, natural text.
Input:
Plain text
qubit is unit of quantum information
qubit can be in superposition
Output:
Plain text
A qubit is a unit of quantum information that can be in a superposition state.
The realizer is responsible for fluency.
It handles:
Plain text
grammar
sentence structure
word choice
flow
compression
readability
natural phrasing
But it is not responsible for truth.
Truth comes from Lucid. The realizer only expresses approved meaning.
Why It Is Called “Denoising”
The canvas is rough and unnatural. The realizer cleans it.
The process looks like this:
Plain text
rough approved text → smoother text → final fluent text
Example:
Plain text
qubit is unit of quantum information
qubit can be in superposition
Becomes:
Plain text
A qubit is a unit of quantum information that can be in a superposition state.
For a physical reasoning example:
Canvas:
Plain text
man is inside or on car
car moves away
man ends up away
Realized output:
Plain text
The man ends up away because he is inside or on the car as it moves away.
What the Realizer Must Avoid
The Denoising Realizer must not make the output “better” by adding unsupported information.
Bad output:
Plain text
The man drives his car away.
Why bad?
Lucid may have approved:
Plain text
man inside_or_on car
car moves away
man at away
But it did not approve:
Plain text
the man is driving
the car belongs to the man
So the sentence is fluent, but unfaithful.
The realizer should prefer simple faithful wording over rich but unsupported wording.
7. Layer 3: Faithfulness Check
Purpose
The Faithfulness Check ensures that the fluent output still matches Lucid’s approved meaning.
This is not an approval gate. Lucid has already approved the answer. This check only verifies that the decoder did not distort the approved packet while rendering it.
The Faithfulness Check answers:
Plain text
Did the decoder add new entities?
Did the decoder add new relations?
Did the decoder add unsupported causal links?
Did the decoder drop important approved claims?
Did the decoder change the meaning of a relation?
Did the decoder exceed the sentence limit?
Did each sentence remain grounded in approved units?
Example: Accepted Output
Approved facts:
Plain text
qubit type_of unit of quantum information
qubit property superposition possible
Generated text:
Plain text
A qubit is a unit of quantum information that can be in a superposition state.
This passes because both claims are directly supported.
Example: Rejected Output
Generated text:
Plain text
A qubit is like a classical bit, but it can be both 0 and 1 at the same time.
This fails because it adds:
Plain text
classical bit
0 and 1
comparison with a classical bit
Those were not approved by Lucid.
Implementation
The Faithfulness Check can be implemented by sending the generated text back through Lucid:
Plain text
generated text → Lucid parse → compare with approved packet
Then compare the extracted meaning against the original approved graph.
If the generated text contains unsupported meaning, reject it or send it back to the realizer for a stricter rewrite.
8. Decoder Input Contract
The decoder should receive only the information needed for rendering.
A clean decoder input could look like this:
JSON
{
  "output_format": "text",
  "max_sentences": 4,
  "approved_units": [
    {
      "unit_id": "definition_qubit",
      "unit_type": "claim",
      "text_intent": "answer",
      "facts": [
        {
          "edge_id": "e1",
          "source": "qubit",
          "relation": "type_of",
          "target": "unit of quantum information",
          "confidence": 0.91
        },
        {
          "edge_id": "e2",
          "source": "qubit",
          "relation": "property",
          "target": "superposition possible",
          "confidence": 0.82
        }
      ],
      "source_refs": [
        "frame_define_qubit",
        "basin_qubit_definition"
      ]
    }
  ],
  "constraints": {
    "forbid_new_entities": true,
    "forbid_new_causal_links": true,
    "require_source_refs_per_sentence": true
  },
  "explicit_omissions": []
}
The decoder does not need the entire internal reasoning tree. It only needs the approved render units and the constraints attached to them.
9. Decoder Output Contract
The decoder should produce final text and, internally, a grounding record.
User-facing output:
Plain text
A qubit is a unit of quantum information that can be in a superposition state.
Internal output:
JSON
{
  "text": "A qubit is a unit of quantum information that can be in a superposition state.",
  "grounding": [
    {
      "sentence": "A qubit is a unit of quantum information that can be in a superposition state.",
      "edge_ids": ["e1", "e2"],
      "source_refs": [
        "frame_define_qubit",
        "basin_qubit_definition"
      ]
    }
  ]
}
The user sees the clean answer.
The system keeps the grounding trace.
This is useful for debugging, audits, and source-backed rendering.
10. Training the Decoder Without Manual Labels
The decoder should not be manually trained with hand-written examples.
Instead, it can learn from open text using Lucid itself.
The training loop is:
Plain text
open text
    ↓
Lucid encodes the text
    ↓
Lucid produces approved render packet
    ↓
Canvas Builder creates rough canvas
    ↓
Denoising Realizer learns:
    canvas + packet → original fluent text
This makes Lucid the automatic teacher.
Training Example
Original text:
Plain text
A qubit is a unit of quantum information that can be in a superposition state.
Lucid-approved facts:
Plain text
qubit type_of unit of quantum information
qubit property superposition possible
Canvas:
Plain text
qubit is unit of quantum information
qubit can be in superposition
Training target:
Plain text
A qubit is a unit of quantum information that can be in a superposition state.
The decoder learns how rough approved meaning becomes fluent language.
Training Data Generation
Pseudo-flow:
Python
for text in open_text_corpus:
    lucid_output = lucid_encode(text)

    if lucid_output.decision != "COMMIT":
        continue

    packet = extract_render_packet(lucid_output)
    canvas = build_canvas(packet)

    train_decoder(
        input={
            "packet": packet,
            "canvas": canvas
        },
        target=text
    )
This avoids manual decoder crafting.
The decoder learns expression patterns automatically from real language.
11. Runtime Flow
At runtime, the decoder works like this:
Python
def decode(lucid_render_packet):
    canvas = build_canvas(lucid_render_packet)

    draft = denoising_realizer(canvas, lucid_render_packet)

    check = faithfulness_check(draft, lucid_render_packet)

    if check.passed:
        return draft

    return repair_or_fallback(canvas, lucid_render_packet)
The normal path is:
Plain text
approved packet → canvas → polished text → check → final answer
If the faithfulness check fails, the system can either:
Plain text
retry with stricter decoding
return a simpler canvas-based answer
or ask Lucid for a safer render packet
The simplest fallback is to return a plain faithful rendering from the canvas.
Example fallback:
Plain text
A qubit is a unit of quantum information. It can be in superposition.
Fallbacks should prefer correctness over beauty.
12. Why This Architecture Is Minimal
The architecture only keeps layers that have a necessary job.
Plain text
Canvas Builder:
Needed because Lucid output is structured, not directly fluent.

Denoising Realizer:
Needed because the user needs polished language.

Faithfulness Check:
Needed because the realizer may accidentally add or distort meaning.
Everything else is removed.
Removed layers:
Plain text
Packet Gate
Meaning Frame Builder
Text Plan Builder
Sentence Planner
Output Selector
Fluency Ranker
Role Assigner
Discourse Planner
These are not needed in the minimal design because Lucid already handles the meaning and commitment side.
The final architecture is intentionally narrow:
Plain text
approved meaning → rough text → fluent text → meaning check
That is the cleanest useful decoder.
13. Design Boundaries
The decoder owns:
Plain text
wording
grammar
sentence flow
compression
readability
surface polish
Lucid owns:
Plain text
understanding
reasoning
approval
safety
world modeling
source validation
omissions
confidence
The faithfulness check owns:
Plain text
meaning preservation
entity preservation
relation preservation
source grounding
claim coverage
The decoder should never cross into Lucid’s responsibilities.
14. Failure Modes
1. Added Meaning
The decoder adds facts that were not approved.
Example:
Plain text
A qubit is like a classical bit...
This adds a comparison not present in the approved graph.
Fix:
Plain text
Faithfulness Check rejects it.
Realizer retries with stricter constraints.
2. Dropped Meaning
The decoder omits an approved claim.
Approved:
Plain text
qubit type_of unit of quantum information
qubit property superposition possible
Bad output:
Plain text
A qubit is a unit of quantum information.
The second claim was dropped.
Fix:
Plain text
Faithfulness Check detects missing edge coverage.
3. Relation Distortion
The decoder changes the relation.
Approved:
Plain text
qubit property superposition possible
Bad output:
Plain text
A qubit always exists in superposition.
This changes “possible” into “always.”
Fix:
Plain text
Faithfulness Check compares relation strength and modality.
4. Over-polishing
The decoder makes the text sound better but less faithful.
Bad output:
Plain text
A qubit is the basic building block of quantum computers.
This may sound natural, but it adds unsupported claims.
Fix:
Plain text
Prefer simple faithful language over rich unsupported language.
15. Final Summary
The decoder is a compact rendering system.
It should not reason, approve, or invent. It should only express Lucid-approved meaning in fluent language.
The final architecture is:
Plain text
Lucid Render Packet
        ↓
Canvas Builder
        ↓
Denoising Realizer
        ↓
Faithfulness Check
        ↓
Final Answer
Each component has one job:
Plain text
Canvas Builder:
Turns approved graph facts into rough text.

Denoising Realizer:
Turns rough text into natural language.

Faithfulness Check:
Ensures the natural language still matches the approved graph.
This keeps the decoder simple, general, and safe.
The guiding principle is:
Plain text
Lucid owns meaning.
Decoder owns expression.
Faithfulness check protects the boundary.
:::