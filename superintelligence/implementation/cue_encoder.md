takes candidates from perception layer and converts them into cue_keys
emits CueCloud with multiple cue_keys. the dmf looks for traces with similar cue_affinities and activates them with learned weights

works in 4 steps:
- turn perception into typed clues
- apply structure rules (clauses and stuff like the words "while")
- look up learned shortcuts
- merge and provide tags that keep ambiguity alive for the rest of the system to work on