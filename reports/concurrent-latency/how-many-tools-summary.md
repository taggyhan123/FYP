# How many tools should we show the model? — short version

A plain-language summary of [`how-many-tools.md`](how-many-tools.md). Same
findings, none of the statistics. Read that one for the evidence.

---

## The question

Our experiments show the model a menu of **4, 16, 64, or 128 tools** and ask it
to pick the right one. Two people looked at that and objected:

> *"A task only needs one or two tools. Why are you showing it 128?"*

> *"No real benchmark shows a menu that big. They show about 5 to 10."*

Both objections are correct on the facts. This document is what happened when
we checked whether they were correct on the conclusion.

---

## Four numbers that are easy to confuse

Everything here gets clearer once these are kept apart:

| | what it means | typical value |
|---|---|---|
| **catalog** | every tool that exists and could be searched | 44,453 |
| **menu (k)** | the tools we actually put in the prompt | 4 / 16 / 64 / 128 |
| **needed** | the tools the task actually requires | ~1.8 (~1.5 in our test tasks) |
| **called** | the tools the model actually calls | ~1 |

The objection is really "menu is much bigger than needed." True. But the menu
isn't supposed to match what's *needed* — the system doesn't know which two
tools a query needs. **That's the entire job retrieval is doing, and it does it
imperfectly.** The menu is a bet on where the right tool probably is.

And a bigger menu doesn't make the model call more tools. It calls about one
per request whatever the menu size — a 32× bigger menu barely moves it. The
task decides how many tools get called, not the menu. (If anything the model
calls too *few*: it needs about 1.5 and calls about 1.)

---

## Why a small menu is not automatically better

If you shrink the menu, you get a cleaner prompt — but you also start missing
the right tool completely.

| menu size | how often the right tool is even in the menu |
|---|---|
| 4 tools | **53%** |
| 16 tools | 68% |
| 64 tools | 82% |
| 128 tools | **86%** |

**At a 4-tool menu, nearly half of all requests are already lost before the
model reads anything.** The correct tool simply isn't there. No model and no
clever ordering can fix that.

So there are two different failure modes, and they are not equally bad:

- **Big menu** — the right tool is there but buried among distractors.
  *Recoverable*: a better model or better ordering can find it.
- **Small menu** — the right tool was never shown.
  *Not recoverable*: nothing downstream can help.

You can see the switch directly in the requests where the model called
nothing at all. With a 4-tool menu, about **6 in 10** of those had no correct
tool to call — declining was the right move, and the search was to blame. With
a 64- or 128-tool menu, **9 in 10** had the right tool sitting in the menu,
unused. **The failure doesn't go away as the menu grows — it moves**, from
"the search didn't find it" to "the model didn't notice it."

---

## The surprise: menu size barely changes the final answer

We measured what actually reaches the user — how often the whole pipeline
(retrieval + model) gets the task right:

| menu size | end-to-end accuracy |
|---|---|
| 4 tools | 20.5 |
| 16 tools | 22.4 |
| 64 tools | 22.0 |
| 128 tools | 21.7 |

**Essentially flat, across a 32× range of menu size.** The two effects cancel:
a small menu picks cleanly but misses often; a big menu almost always has the
tool but the model gets distracted.

**This means menu size can't be argued on accuracy at all.** It has to be
chosen for other reasons — speed and cost (favour small), or matching what real
deployments actually do (favour large).

---

## What real systems actually do

Two different worlds, and they disagree:

**Benchmarks show 5–10 tools.** But they get there using a *router* — a
component that re-searches the catalog on every single query and hands over
only the best few. One benchmark had to build a purpose-made router to manage
its 527 tools.

**Real MCP deployments mostly have no router.** They just inject everything.
One well-scoped server is 20–30 tools — fine. But connect four or five ordinary
integrations (GitHub, Slack, a database, a calendar) and you're at **~150
tools**. Cloudflare disclosed a case with over a million tokens of tool
definitions. Cursor had to impose a hard 40-tool cap because overload became a
real operational problem.

**So both regimes are real.** Benchmarks measure the routed world. Our k=64/128
measures the un-routed one — which is where most MCP users actually live. The
mistake would be pretending they're the same problem.

---

## The uncomfortable finding

At the benchmark's menu size, **our headline effect disappears.**

At 10 tools, every ordering policy we test finishes in the same time —
66 to 68 milliseconds, all six of them. The cache benefit that ordering
provides is real, and at that menu size it is worth nothing, because the prompt
is too short for it to matter.

**Had we used the standard menu size from the start, we would have measured
nothing at all.** That's worth saying plainly.

---

## The finding that rescues it

Then we tried the design that matches how routed systems actually work:

> **Search 64 tools. Reorder them. Show only the top 10.**

This is the realistic setup — a router picking a shortlist from a bigger pool.
And here the policies separate sharply, on something we hadn't been measuring:

| policy | how often the right tool survives into the shown 10 |
|---|---|
| no reordering | 62% |
| **ToolTrie-v1** | **62%** — unchanged |
| ContextPilot | **47%** |
| ToolTrie-v0 / alphabetical | **8%** |

**ContextPilot pushes the correct tool out of the visible window in one request
in seven.** It reorders to win cache reuse, and when only the top 10 survive,
some of what it promoted displaces what the user actually needed.

**ToolTrie-v1 doesn't.** It only moves tools it has direct evidence about and
leaves everything else in the retriever's order — so when the list gets cut,
the important things are still near the top. It gets **4.3× the cache reuse and
13% faster responses, and loses nothing.**

ToolTrie-v0 and alphabetical are the cautionary tale: they achieve good cache
reuse by sorting tools into an order that has nothing to do with relevance.
**92% of requests lose the right tool.** Cache maximised, system broken.

### We checked this on a second search engine, and it got worse for ContextPilot

The result above used one way of searching the catalog. Repeating the whole
thing with a completely different search method:

| policy | survives (engine A) | survives (engine B) |
|---|---|---|
| no reordering | 62% | 59% |
| **ToolTrie-v1** | 62% | **60%** — slightly *better* than not reordering |
| ContextPilot | 47% | **40%** |
| ToolTrie-v0 / alphabetical | 8% | 12% |

**So it isn't a quirk of one setup.** On the second engine ContextPilot throws
away **one request in five** rather than one in seven, and ToolTrie-v1 comes
out marginally *ahead* of doing nothing at all.

### And it gets worse for ContextPilot when requests arrive in bursts

Real traffic comes in clusters — related requests close together. We tested the
most clustered arrival order our tasks allow, which is exactly the kind of
traffic ContextPilot was designed for. It made ContextPilot's problem **worse**:
it now throws away the right tool about **twice as often** as before, while
ToolTrie-v1 loses only a little.

The reason turns out to be simple, and it explains every result in the
project:

- **ContextPilot looks for tools that *every* request shares**, and moves those
  to the front. That works brilliantly when such tools exist — like the padded
  menus, where 63 of 64 tools are in every request.
- **ToolTrie-v1 looks at what the *previous* request used**, and reuses that
  order. That works when requests resemble their neighbours, even if nothing is
  shared by everyone.

**Real retrieved requests never share a common core** — even at their most
clustered, no tool appears in every request. They only resemble their
neighbours. That is ToolTrie-v1's situation, not ContextPilot's.

---

## What this changes

We had been describing ToolTrie-v1 as *"the policy that gets the most cache
reuse."* That framing turns out to be almost beside the point:

- At realistic menu sizes, **cache reuse buys nothing measurable**.
- What actually separates the policies is whether reordering for the cache
  **damages the results the user sees**.
- **ToolTrie-v1's real property is that it is safe** — it takes the cache
  benefit without disturbing what retrieval worked out.

That is a different, and more useful, claim than the one we started with.

---

## The one-sentence version

Menu size doesn't decide accuracy, ordering doesn't decide speed at realistic
menu sizes, and the thing that actually matters is whether reordering for the
cache survives having the list cut short — which ToolTrie-v1 does and
ContextPilot does not.

---

## Honest limitations

- Tested on **200 tasks** and **single-turn** requests only. Real agents are
  multi-turn, where errors compound. (The main result *was* checked on a
  second search engine and held up — see above.)
- The 200-task sample we used is **easier than average** — it happened to
  contain more single-tool tasks than a random draw would. We re-ran the main
  64-tool comparison on a random sample, which turned out *harder* than
  average. Scores dropped — by about a third for the best policies — and the
  gaps between policies narrowed. **ToolTrie-v1 still beat ContextPilot**, by
  about half as much. But ToolTrie-v1's small lead over no reordering became a
  tie, and ContextPilot slipped behind the simple `frequency` policy. So the
  main comparison holds; the absolute scores, and the finer orderings between
  similar policies, should not be trusted.
- One policy, `frequency`, looked far better than it was until we measured
  end-to-end. **Which metric you pick can reverse the answer**, which is why
  the full document is careful about it.
