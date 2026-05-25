
## Normal backprop

A normal neural net learns with two different phases:

```
1. Forward:   input goes through the model → answer2. Backward:   error is sent backward through the model → update weights
```

So if the model says:

```
cat
```

but the right answer is:

```
dog
```

backprop asks:

```
Which weights throughout the whole network caused the wrong answer?
```

Then it sends correction signals backward through the whole system.

That is powerful, but expensive and global.

---

# What Equilibrium Propagation does

Equilibrium Propagation trains a system more like this:

```
1. Let the system settle naturally.2. Slightly nudge the output toward the right answer.3. Watch how the internal state changes.4. Update local connections based on that difference.
```

Imagine a marble rolling into a valley.

## Free phase

You give the system an input.

```
input: picture of dog
```

The system settles into an answer valley:

```
system settles → “cat”
```

That is the **free phase**.

## Nudged phase

Now you slightly pull the output toward the correct answer:

```
nudge output toward “dog”
```

The whole system shifts a little.

Now compare:

```
before nudge: internal state Aafter nudge:  internal state B
```

The learning update comes from:

```
how each local part changed between A and B
```

So instead of a separate backward pass, the system learns from the difference between two settled states.

---

# Why that is different

Backprop:

```
forward answer→ compute error→ send gradient backward through separate backward computation
```

EqProp:

```
settle normally→ nudge output→ settle again→ local differences update connections
```

So the big difference is:

```
Backprop uses an explicit backward error path.EqProp uses the system’s own dynamics to carry the correction.
```

That is why people say it is more “local.”

Each part can update based on:

```
my activity before the nudgemy activity after the nudgemy nearby connection
```

It does not need the whole global backward graph.

---

# Simple analogy

Imagine a crowd trying to arrange itself into a shape.

## Backprop version

A boss looks at the final shape and tells every person exactly how they should have moved.

```
“You, move left.”“You, move back.”“You, turn 12 degrees.”
```

That is global instruction.

## EqProp version

The crowd first forms a shape.

Then you gently pull the final edge toward the desired shape.

People locally adjust because nearby people moved.

Each person learns:

```
when the goal nudged the group,how did my local position need to change?
```

No boss sends a full backward instruction to everyone.

---

# What Zyphra’s work changes

Old EqProp mostly worked cleanly for special systems called **energy-based models**.

Those are systems that naturally settle into stable low-energy states.

Zyphra is saying:

```
We extended this kind of learning to more realistic neuron-like dynamical systems.
```

Specifically, they use FitzHugh–Nagumo neurons, which are simplified models of spiking/excitable biological neurons. Their paper says they extend Equilibrium Propagation to skew-gradient systems and apply it to diffusively coupled FitzHugh–Nagumo networks.

So the new thing is:

```
EqProp may work beyond clean energy-based models,in messier neuron-like systems.
```

That matters because real brains and future hardware are not clean transformer layers.

---

# Why this matters for your architecture

Your architecture has something similar to “settling”:

```
cue activates traces→ interference shapes energy→ basins compete→ system settles into a basin→ lucidity checks it
```

So instead of training with:

```
global backprop through perception/cue/DMF/binding/context/interference/basins/lucidity/decoder
```

you might train parts of it with:

```
1. Let basin system settle.2. Lucidity says the result should be nudged toward better state.3. Let system settle again.4. Update local trace-basin-interference links based on the difference.
```

That is the connection.

For your system:

```
free phase:    system settles into wrong basinnudged phase:    lucidity/projector pushes toward correct basinlocal update:    traces/interference links/basin links adjust
```

That could be much cheaper than training everything globally.

---

# What it does not do

It does **not** magically solve:

```
languagereasoningAGIASIworld modeling
```

It is not an architecture by itself.

It is a **learning rule**.

It answers:

```
How can a settling system update itself locally?
```

Not:

```
What should the system think?
```

Your architecture still needs:

```
tracesbindingcontext-opinterferencebasinsluciditydecoderorchestrator
```

EqProp would only help train the dynamical parts.

---

# The simplest summary

Backprop:

```
answer wrong→ send error backward through the whole network→ update weights
```

EqProp:

```
answer wrong→ gently nudge the answer toward correct→ system shifts→ update local connections from the shift
```

Zyphra’s contribution:

```
This local-learning idea can work in more realistic neuron-like dynamical systems than before.
```

Why you care:

```
Your basin/trace system is also a settling dynamical system,so this could be a useful way to train DMF, interference, and basins locally.
```