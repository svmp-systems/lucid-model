It basically gives the chat a memory. 

It remember things in the same session but does not share any memory with other new sessions. Here sessions are basically new chats. t keeps full audit history on disk. It send only relevant/recent context into the pipeline. It supports simple recall and updates like 'remember blue' then 'make it green' by using rebinding. It ofcourse also has turn-by-turn recall. It runs the same pipeline and does not create a seperate chatbot path.

for example: I create a new chat, a new session is now created
I ask it to remember the colour blue, this is one trun
now next turn i ask what colour it replies with blue
now if i tell it to remember green in the same session rebinding is done
now in my next turn i ask what colour it replies with green
now if i start a new session ie new chat and then ask wha colour
its gona reply with it has no colour assigned to
this is because new session hav no relation with the previous sessions

It also sumarises the previous texts for the model if a session is too long but here the full audit log is saved
