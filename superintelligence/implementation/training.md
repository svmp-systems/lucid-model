training exists by producing synthetic corpora using generator and then running pipeline via pipeline-orchestrator. the training orchestrator runs pipeline and validates output then calls on governor to decide whether any changes need to be made. if a change has to be made, it chooses the smallest possible magnitude of change and then runs it on a shadow-test. if it passes 3 times then the change is commited. if it fails then the next smallest magnitude update is made.

highly optimized via:
- a training governor that decides whether something should be updated, deferred or left untouched
- a "no-update" on high margin success
- module targetted updates
- a decoder-only correction path that is used when system was right but output was wrong

also writes fully machine and human readable training logs.