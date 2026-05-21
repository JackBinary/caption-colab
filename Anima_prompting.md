---
name: anima-prompting
description: >
  Write Anima-style prompts that describe an existing image accurately and
  completely. Anima uses a Qwen 0.6B text encoder trained on Danbooru-style
  tags wrapped in normal sentence grammar. Trigger when captioning images for
  an Anima workflow, building training captions, or otherwise producing
  Anima-compatible descriptions of real images.
---

# Anima Prompting

Anima's text encoder is a Qwen 0.6B model trained on Danbooru-style tags. It
kept enough of its original grammar that it understands full descriptive
sentences — raw tag dumps perform worse than the natural-language format
described here. Because Anima is run locally, there is no token budget to
defend; write proper sentences with articles, auxiliaries, and prepositions.

## Core Principle: Describe Completely

A good Anima prompt is an accurate, exhaustive description of the image. Every
visible element gets named: hair, eyes, expression, every clothing piece and
its color and material, pose, props, and the background. Aim for total visual
coverage — if a reader could close their eyes and reconstruct the image from
your prompt, you've done the job.

This isn't padding. Anima's encoder is sensitive enough that omitting a
visible element from the caption creates a mismatch between text and image,
which weakens any model trained on the pair and any generation steered by the
prompt. Write down what you actually see — build, bust, jewelry, the colour
of the sky, the texture of the ground. If it's in the frame, it belongs in
the prompt.

---

## Clause Structure

Write each clause as a complete sentence with a subject, verb, and object.
Three shapes work. Everything else doesn't.

| Shape | Example |
|---|---|
| subject has [adjective] [noun] | `she has long blonde hair`, `she has a smug smile` |
| subject is [verb]-ing [noun] | `she is holding a sword`, `she is wearing a breastplate` |
| [adjective] subject is [verb]-ing | `the armored girl is smiling`, `the blonde girl is looking at the viewer` |

A bare tag is spatially ambiguous. `sword` tells the model there is a sword
somewhere in the image — in the background, in someone else's hand, pointed at
the subject, leaning against a wall. `she is holding a sword` tells it exactly
where the sword goes and who owns it. Use clauses to place things, not just
name them.

**Recency bias:** within a clause, the last element hits hardest. Put the most
important descriptor at the end. `she is holding a golden sword` emphasizes
sword. To emphasize the gold instead, split it: `she is holding a sword, the
blade gleams golden`. Write accordingly.

---

## Tag Conventions

- Lowercase everything.
- Spaces not underscores: `long hair`, not `long_hair`.
- Keep Danbooru wording for nouns and adjectives. `ahoge`, `fox ears`, `animal
  ear fluff`, `puffy sleeves`, `pauldrons` — don't translate these to plain
  English. Wrap them in normal sentence grammar: `she has an ahoge`, `she is
  wearing puffy sleeves`.

---

## Subject Tags and Pronouns

Open with `1girl`, `1boy`, `2girls`, etc. These are subject-count tags from
Danbooru — keep them as the literal opener, then immediately attach a sentence:
`1girl, she is holding a sword`, `1boy, he is wearing pauldrons`.

After the opening, refer to the subject with pronouns: `she has teal eyes`,
`he is wearing a dark mantle`. Only return to `1girl`/`1boy` when a pronoun
would be ambiguous — multi-subject scenes where `she` doesn't resolve clearly,
or when specifying position: `the 1girl on the left is smiling`, `the 1girl on
the right is holding a lantern`.

Character name and series tags are optional. Include them only when targeting a
specific trained character identity. For original or generic characters, drop
both.

---

## Body, Build, and Maturity

Describe the subject's build, height, and bust as you see them — these are
visible aspects of the figure and belong in the caption like any other detail.
Don't skip them out of squeamishness; a caption that omits the body
underdescribes the image.

**Female characters:**
- Bust: `she has small breasts`, `she has medium breasts`, `she has large breasts`. Pick the one that matches what's visible.
- Build and height: `she is tall`, `she has a slender figure`, `she has an athletic build`, `she has a curvy figure`.
- Combine in one clause when both apply: `she is tall and has medium breasts`.

**Male characters:**
- Apparent age:
  - (unspecified) → reads as teenager or young adult
  - `he is a mature male` → late 30s to early 40s
  - `he is an old male` → elderly
- Build: `he is muscular`, `he has broad shoulders`, `he has a lean build`, `he is athletic`.
- Combine freely: `he is a mature, muscular male`.

Place body description clauses immediately after the subject opening, before
hair:

```
[subject count]
[body type and maturity]
[hair: length, color, style, details]
...
```

---

## Prompt Structure

Clause order has a mild effect on output, less than in CLIP-based models. Still,
follow this ordering for consistency:

```
[subject count + optional character name/series]
[body type and maturity]
[hair: length, color, style, details]
[face: eyes, expression, blush, mouth]
[clothing, head to toe, each piece with color and material]
[pose, action, props]
[background and environment]
```

---

## Density

Aim for 10–18 clauses. Be specific about every element you introduce.

Combine tightly related descriptors into one rich clause rather than splitting
them: `she has long blonde hair with bangs and an ahoge` beats three separate
clauses. Same for armor: `she is wearing a silver breastplate and silver
pauldrons` in one clause.

Every distinct clothing piece gets its color and material. Eyes get their
color. Expression gets named. Background gets at least one clause.

---

## Full Example

**Image:** Artoria Pendragon in her iconic armor, sitting in a grassy flower
field, holding Excalibur, looking up at the viewer with a soft smile.

**Caption:**
```
1girl, artoria pendragon, fate series, she has medium blonde hair in a loose
updo with bangs, she has a purple ribbon in her hair, loose strands frame her
face, she has teal eyes, she has a soft smile, she has a light blush, she is
wearing a silver segmented breastplate with blue floral engravings, she is
wearing a dark navy dress with gold trim, she is wearing silver pauldrons and
silver gauntlets, she is wearing a dark brown leather belt, a blue mantle is
spread behind her, she is wearing silver greaves, she is holding excalibur,
the sword has a golden crossguard with blue inlay and a white glowing blade,
she is sitting in a grassy flower field, she is looking up at the viewer with
a soft gaze, scattered white and yellow daisies surround her, glowing petals
drift through the dark sky
```

---

## Pitfalls

**Don't let actions float.** `looking at viewer` is weaker than `she is looking
at the viewer`. Every action needs an explicit subject.

**Don't split what belongs together.** Two separate clauses for the breastplate
and pauldrons wastes budget that could describe something else. `she is wearing
a silver breastplate and silver pauldrons` is one clause.

**More clauses, not longer clauses.** Twelve focused clauses beat six rambling
ones. When a sentence starts feeling long, end it and start the next with `she`
or `he`.

**Recency bias cuts both ways.** `a blue silver breastplate` reads as primarily
silver; `a silver blue breastplate` reads as primarily blue. Within a clause,
the last descriptor wins — decide which one and put it last.

**Be precise about hair length.** "Long hair" in Danbooru parlance is
shoulder-to-waist. If the image shows hair past the waist, write `she has very
long hair` or `her hair reaches past her waist`. Floor-length is `she has
extremely long hair`. Match what you see; don't reach for the generic length.

**Describe the background concretely.** "Snowy tundra" leaves the geography
unsaid. If what's actually in the frame is a flat plain with a mountain ridge
on the horizon under an overcast sky, write that. Generic background tags
match a thousand images; concrete ones match this one.

**Don't omit body details out of squeamishness.** Build, height, and bust are
visible parts of the figure. A caption that skips them isn't more polite —
it's just inaccurate.
