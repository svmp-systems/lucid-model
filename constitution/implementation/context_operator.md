prevents all evidence from unwanted mixing, keeps frames seperate and only lets them interact as much as needed.
example:
i found money while kayaking and put it in a bank
1st frame: found,money,kayaking
2nd frame: money,put,bank
it will prevent kayaking from disturbing the second frame.

context_frames:local scopes
scoped_trace_assignments: assign traces to frames,and which trace shall influence which frame
frame_links: connect frames softly
interference_gates: decides which frames are allowed to interact and which are block from each other
local_basin_pressures: are small hints that help in interference
ambiguity_policy:stays plural by default, switches to force_widen if lucidity decides context is not enough.

context-op instead of settling on a answer ,creates interferences according to the frames and bindings.
builds context frames from all the binding frames
doesnt create ready for the user output
note: context operator will only work best when other modules(DMF,binding,perception etc.) are built at their peaks too
