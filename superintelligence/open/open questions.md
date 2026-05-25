1. 
Learned trace IDs (t0142) that gain meaning through use is elegant — but it's the same hard problem as "how does unsupervised structure emerge?" Transformers cheat by absorbing structure from massive data. You need either:


A training recipe that actually grows traces/basins/interference links, or
A hybrid bootstrap (start from an existing encoder, distill into traces)



2. 
How to solve continual learning in a model where  memory and reasoning is a pipeline and memory is an external storage 


3. 
The training governor idea is good--but responsibility classification is notoriously hard. "Binding error vs basin collision vs decoder error" is easy to write, hard to measure reliably.

