Lucidity collapses the multiple hypothesis that are maintained throughout the pipeline by lazy collapse into structred graph. it picks a single approved internal state. 

it outputs decision, decoder_policy, check_results and confidence_summary 
depending on the decision it also picks committed_state, preserved_hypothesis, search_directives, render_packet

render_packet is used to give script to decoder. it is built inside lucidity by approved units, alternatives, omissions and faithfulness rules. decoder reads the render_packet only 

lucidity checks 9 things 
- margins -- is top basin clearly ahead of 2nd highest? dont fake certainty if not 
- coverage -- does winning theory accoutn for hte important evidence ? 
- coherenve - no incompatible roles in the same frame 
- scope -- checks if contextop frames were respected 
- projector fit -- if grid/plan was simulated, does it match input and expected output ? 
- contradictions -- are there any serious interference conflicts inside this frame? 
- maturity -- checks if we're commiting unverified memory (from continual learning)
- risk -- combines task + stakes.

it can have multipel outputs -- commit, preserve_ambiguity, request_projection, search_wider and recheck_binding

inside commit, theres commit_shape which determines whether its single frame, per frame, assembled frames, rollot plan, etc,. 
