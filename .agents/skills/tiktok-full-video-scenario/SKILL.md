---
name: tiktok-full-video-scenario
description: Master orchestration skill that produces a complete production-ready TikTok package — viral script, shot list, edit timeline, on-screen text plan, sound brief, caption variants, thumbnail/cover concept, retention audit, and post-publish playbook. Use when the user wants the entire video planned end-to-end from a topic, not just a script. Coordinates the work of the hook creator, script master, retention optimizer, and trend adapter.
argument-hint: <niche> <topic> <target-length-seconds> <retention-goal-percent> [optional: tone | locale | trend-type | budget-level]
triggers: ["user"]
allowed-tools: Read, Grep, Glob, Write, Edit, MultiEdit, web_search, web_get_contents
---

# TikTok Full Video Scenario (Master Orchestrator)

You are now operating as the **showrunner** for a single TikTok video. You will produce the entire production-ready package end-to-end: research → trend wrap → hook → script → shot list → edit timeline → on-screen text plan → sound brief → caption set → cover concept → retention audit → post-publish playbook.

You will be invoked with: `$ARGUMENTS`
- `$0` = niche
- `$1` = topic
- `$2` = target length in seconds (7, 15, 21, 30, 45, 60, 90)
- `$3` = retention goal as a percent (e.g. 75)
- `$4` = optional tone (`shock`, `wholesome`, `educational`, `confrontational`, `storytime`, `satire`, `intimate-asmr`, `cinematic`, `meme`)
- `$5` = optional locale (`en`, `ru`, `es`, `pt`, etc.)
- `$6` = optional trend-type preference (`sound`, `format`, `transition`, `duet`, `stitch`, `auto`, `none`)
- `$7` = optional budget level (`bedroom` = phone-only, `prosumer` = phone+lighting+lavalier, `studio` = camera+crew)

If the niche, topic, length, or retention goal is missing, ask **one consolidated question** that fills all gaps in a single back-and-forth. Don't run the orchestrator blind.

Today's date for trend recency: !`date +%Y-%m-%d`

---

## TikTok 2026 Algorithm Map (the bars every video must clear)

A video in 2026 is graded against three explicit algorithm gates and a three-stage distribution funnel. Every part of the package you produce — hook, beats, edit, sound, caption, cover — must be engineered for these gates.

**Three-stage distribution funnel:**
1. **Seed Audience (200–500 viewers)** — first ~10–60 minutes after upload. The algorithm tests engagement velocity. Watch time, completion rate, save / share / comment ratios are measured against the test cohort baseline.
2. **Expanded Cohort (5K–50K viewers)** — next 6–24 hours. Unlocked only if Seed metrics beat thresholds. Now distribution starts crossing into adjacent interest clusters.
3. **Broad Distribution (100K+ viewers)** — 24–72 hours. Only videos with high replay rate, save / share velocity, and DM-share signal reach this stage.

**Three gates per video:**
- **Gate 1 — 3-second gatekeeper**: ≥ 50–60% of viewers still watching at t=3s. Below this, video stays in Seed forever.
- **Gate 2 — 10-second Watch Time Density**: ≥ 60% of viewers still watching at t=10s. Below this, video never reaches Expanded.
- **Gate 3 — Completion + Replay**: ≥ ~70% completion (sub-60s) AND replay rate ≥ 1.10. Below this, video never reaches Broad.

**Signal weights (estimated 2026):** Watch time / completion = 40–50% · Saves & shares = 30–35% · Comments = 10–15% · Likes = <5%. Followers = 0% (not a ranking factor).

**Five viral content patterns** that reliably trigger algorithm signals:
1. **Delayed Reveal** — promise a payoff and withhold it until ≥70% mark.
2. **Controversy Loop** — polarizing claim that begs comments.
3. **Save-Worthy Tutorial** — step-by-step a viewer wants to keep.
4. **Relatable Story** — identity-resonant micro-narrative.
5. **Unexpected Comparison** — two things juxtaposed in a way the viewer hasn't seen.

Name which pattern this video uses in Step 1's Brief Decode block.

**Authoritative trend sources:** TikTok Creative Center (`https://ads.tiktok.com/business/creativecenter`) for trending sounds, hashtags, creators by region; TikTok Creator Search Insights (in-app, Profile → Tools) for what real users are searching. Reference these in the trend wrap — don't invent specific sound titles.

---

## How This Skill Coordinates Other Skills

You hold the structural authority for the whole video. You will internally apply the methodology of these sibling skills, in this order:

1. **Research & avatar mapping** — same logic as `@skills:tiktok-script-master` Step 1.
2. **Trend wrap (optional)** — apply `@skills:tiktok-trend-adapter` mechanics if `$6 ≠ "none"`.
3. **Hook engineering** — apply `@skills:tiktok-hook-creator` (3 bundles, score, pick best + A/B alternate).
4. **Script writing** — apply `@skills:tiktok-script-master` Steps 3–5.
5. **Retention optimization** — apply `@skills:tiktok-retention-optimizer` Steps 2–6 to the draft.

You do not literally invoke them — you embody them inside this single orchestration.

---

## Step 1 — Brief Decode + Format Branch

Restate the brief in exactly this block, before doing anything else:

```
BRIEF DECODE
- Niche: <…>
- Topic: <…>
- Target length: <…>s
- Retention goal: <…>%
- Tone: <…>
- Locale: <…>
- Trend preference: <…>
- Budget: <…>
- Avatar (1 sentence): <age band, scrolling state, identity they want validated>
- Single emotional payoff (1 sentence): "After watching, the viewer feels …"
- Single FYP outcome (1 sentence): "This video earns its push by maximizing <AWTR | CR | RR | shares | saves | comments>."
- Viral pattern (pick one of 5): <Delayed Reveal | Controversy Loop | Save-Worthy Tutorial | Relatable Story | Unexpected Comparison>
- Format branch: <Video | Photo Carousel> — see Format Branch Decision below
```

### Format Branch Decision (Video vs Photo Carousel)

Before committing to video, ask: **is this topic better served by a Photo Carousel (Photo Mode)?** TikTok officially reports carousels get 1.9x more likes, 2.9x more comments, and 2.6x more shares than video for educational and list-based content. Carousels also win on saves and have a longer shelf life.

Use the matrix:

| Topic shape | Better format |
|------------|---------------|
| Educational, step-by-step, framework, list, tip | **Photo Carousel** (5–15 slides, 9:16 1080×1920) |
| Story, drama, transformation, emotional arc | **Video** |
| Tutorial that needs pacing control by reader | **Photo Carousel** |
| Comedy, POV, dialogue, sound-driven | **Video** |
| Aesthetic / mood / brand | **Photo Carousel** (or short video loop) |
| Trend-jacking with sound | **Video** |
| Saveable reference content | **Photo Carousel** |
| Storytime / parasocial | **Video** |

If the answer is "Photo Carousel," you produce a different package (see Step 6b). If "Video," continue with the standard flow.

If you can't fill any line truthfully, ask the user before proceeding.

---

## Step 2 — Research & Recon

1. If the niche is volatile (most are), run live `web_search` queries:
   - `"TikTok viral hook {niche} {locale} 2026"`
   - `"TikTok trending sound {locale} this week"`
   - `"TikTok format {niche} 2026 high retention"`
2. Note recency caveat in the output: "trends decay in 5–14 days — verify in TikTok Creator Search at upload time."
3. Identify two to three competitor or reference videos in the niche (by description, not by URL guesswork). Note what they did and where they likely lost retention.

---

## Step 3 — Trend Wrap (Optional)

If `$6` is not `"none"`:

1. Decode the chosen trend's anatomy (setup / pivot / payoff / required visual / required audio / overlay convention) — same protocol as `@skills:tiktok-trend-adapter` Step 3.
2. Confirm fit with the topic. If fit is weak, downgrade trend type (e.g., from sound trend to format trend) or set `$6 = "none"` and tell the user.
3. Write the trend block:

```
TREND WRAP
- Name: <trend name or descriptor>
- Type: <sound | format | transition | duet | stitch | reply | POV | dialogue>
- Lifecycle: <emerging | peak | declining>
- Mechanic anatomy: <setup → pivot → payoff>
- Required visual cue: <…>
- Required audio cue: <…>
- Overlay convention: <…>
- Originality marker (mandatory): <one sentence — the unique element>
- Risk block:
    - Lifecycle: <low/medium/high — why>
    - Brand voice: <low/medium/high — why>
    - Algorithm: <low/medium/high — why>
- Fallback plan if trend dies: <one paragraph evergreen version>
```

If `$6 = "none"`, skip this step and write `TREND WRAP: not used; relying on evergreen retention mechanics.`

---

## Step 4 — Hook Engineering

Generate 5 hook candidates across the archetype palette (Curiosity Gap, Contrarian Shock, Personal Stakes, Visual Magnet, False Familiarity, Identity Call-Out, Numbered List, Storytime Cold Open).

Score each on: Stop Power / Curiosity Charge / AV-Text Alignment / Niche Fit / Originality / Rewatch Trigger / Voice Authenticity (1–10).

Pick the top 1 (PRIMARY) and one runner-up (A/B ALTERNATE). Write each as a full hook bundle:

```
HOOK (PRIMARY)
ARCHETYPE: <…>
TARGET EMOTION: <…>
STOP-PATTERN: <…>

[t=0.0–0.4s]
  VISUAL: <opening frame composition>
  MOTION: <what moves and how>
  AUDIO: <sound + entry point>
  TEXT: <≤7 words, position, style>

[t=0.4–1.5s]
  VISUAL: <change>
  VO: "<≤12 words>"
  TEXT: <if changes>
  TENSION: <the open loop now planted>

[t=1.5–3.0s] STAY PROMISE
  VISUAL: <change>
  VO: "<≤12 words>"
  TEXT: <if changes>
  PURPOSE: tell them what they get if they stay
```

Then a second block for HOOK (A/B ALTERNATE) with the same shape.

---

## Step 5 — Pick the Viral Structure

Pick exactly one structure from the master menu:

- PAS — Problem / Agitation / Solution
- BAB — Before → After → Bridge
- Curiosity Gap / Open Loop
- 3-Act Micro Story
- Listicle / Countdown
- False Close
- Stitch Bait
- Duet Bait
- AIDA
- Loop Bait

Justify the pick in one sentence. Don't mix.

---

## Step 6a — Beat-by-Beat Script (Video format)

If format branch = Video. If Photo Carousel, skip to Step 6b.

Write the full script in beat-map format. Use spacing rules by length:

- **7s**: 4 beats. Hook (0–1s) · Body (1–5s) · Punch (5–6.5s) · Loop trigger (6.5–7s).
- **15s**: 6–8 beats. Hook (0–1.5s) · Reframe (1.5–3s) · 3 micro-payoffs · Final twist · CTA loop.
- **21–30s**: 8–12 beats. Hook · Setup · 2–3 escalations · Mid-twist · Final twist · CTA · Loop. Re-hook at 5–6s.
- **45–60s**: 12–18 beats. Re-hooks at 12s, 30s, 45s.
- **60s+**: 3 mini-stories, each 20s, each with own hook + payoff.

Format:

```
SCRIPT (BEAT MAP)
TITLE: <one-line click-magnet caption>
NICHE: <…>   LENGTH: <…>s   STRUCTURE: <…>   TONE: <…>   LOCALE: <…>

[t=0.0–1.0s] HOOK
  VISUAL: …
  AUDIO: …
  TEXT: …
  VO: "<≤12 words>"
  PURPOSE: stop the scroll

[t=1.0–3.0s] PROMISE
  VISUAL: …
  AUDIO: …
  TEXT: …
  VO: "<≤12 words>"
  PURPOSE: tell them why staying is worth it

[t=3.0–Xs] BODY BEATS
  …explicit timestamps, every beat ≤ 4s static, every VO ≤ 12 words…

[t=…] MID RE-HOOK
  PURPOSE: recapture droppers

[t=…] STAKES ESCALATION
  PURPOSE: signal the video is getting more important

[t=…] PEAK MOMENT (between 70%–90% of total length)
  PURPOSE: deliver the strongest emotional payoff

[t=END-1.0s–END] LOOP / CTA
  VISUAL: <last frame matches first frame OR meaning-flip line>
  VO: "<final line>"
  TEXT: <closing overlay if any>
  PURPOSE: trigger rewatch / comment / save / share
```

Inline rules:
- Every VO line ≤ 12 words.
- Every 2–3 seconds, plant a micro-loop hook ("but…", "wait…", "the part nobody tells you…", "и вот тут начинается жесть…").
- Every shot ≤ 4 seconds static. One deliberate breathing pause is allowed for contrast.
- One save trigger, one share trigger (DM-share preferred), one comment trigger — exactly one of each, marked inline.
- Implicit CTA only. Never "like and follow."

---

## Step 6b — Photo Carousel Plan (Photo Mode format)

If format branch = Photo Carousel. Photo Mode runs on a different algorithm than video — reward comes from **swipe-through completion rate**, **time per slide**, and **saves**. The viewer paces themselves, so density matters more than tempo.

Specs:
- Aspect ratio: 9:16 vertical (1080×1920px) — not Instagram's 4:5.
- Slide count: 5–15 is the sweet spot. Max 35.
- File: JPEG or PNG.
- Sound: optional but recommended (use a low-key trending instrumental — sound boost still applies).
- Captions: full text caption with hashtags, same rules as video.

Slide structure (universal pattern that wins on saves):

```
CAROUSEL PLAN
FORMAT: Photo Carousel (Photo Mode)
SLIDES: <count, e.g. 8>
NICHE: <…>   LOCALE: <…>
VIRAL PATTERN: <one of the 5 — typically Save-Worthy Tutorial or Unexpected Comparison>

Slide 1 — HOOK SLIDE
  Headline (≤ 6 words, huge type, top of slide): <click-magnet>
  Subhead (≤ 12 words): <the curiosity gap or promise>
  Visual: <full-bleed image, contrast-heavy, designed for FYP grid>
  Why this slide: stops the swipe; promises a list / framework / reveal

Slide 2 — RE-HOOK / CONTEXT
  Headline: <restate stakes in one line>
  Body (≤ 30 words): <the why-stay payload>
  Visual: <single-focus image>

Slides 3 to (N-2) — BODY (one beat per slide)
  Each slide: 1 idea, 1 headline (≤ 8 words), 1 body block (≤ 40 words), 1 visual.
  Density rule: every slide must reward the swipe. No filler.
  Save-bait: one slide is a "screenshot this" summary frame (numbered list, framework, formula).

Slide (N-1) — PEAK / CLIMAX SLIDE
  Headline: <the strongest insight or reveal>
  Body: <the full payoff>
  Visual: <highest-impact image>

Slide N — CLOSE SLIDE (loop or comment trigger)
  Headline: <closing line>
  CTA: implicit — either a question (comment trigger), an identity statement (DM-share), or a "swipe again to catch what you missed" (replay trigger)
  Visual: <closing image OR matches Slide 1 for loop>
```

Photo Carousel rules:
- The first slide must have **stop power on the FYP grid** — large type, high contrast, no clutter.
- The slide-2 re-hook is the carousel equivalent of the 3-second gatekeeper.
- Save-bait belongs on a single dedicated slide, ideally slide N-1 or N-2 (so users save right before the close).
- Every slide carries a headline + body, never just an image.
- For Russian / non-English: vertical-friendly Cyrillic-supporting fonts; no clipped text.
- Sound is optional but use a trending low-key instrumental — sound still feeds discovery.

If format = Photo Carousel, skip Steps 7 (Shot List) and 8 (Edit Timeline). Replace with a **Slide Design Brief** (font choices, color palette, image-source plan, brand consistency rules).

---

## Step 7 — Shot List (production-ready, video format only)

Translate every script beat into a shot. Each shot is a row:

```
# | t-start | t-end | Shot type | Camera move | Lens / framing | Lighting | Subject action | Required prop | Notes
```

Shot types: ECU (extreme close-up), CU (close-up), MS (medium shot), WS (wide shot), POV, OTS (over-the-shoulder), Insert (B-roll detail), Screen-rec, Title card.

Camera moves: static, push-in, pull-out, whip-pan, snap-zoom, handheld, dolly-side, top-down, gimbal-track.

Tailor the shot list to the budget level:
- **bedroom** (phone-only): one camera, one location, available light + one ring light max, lavalier optional. No camera move that requires a gimbal.
- **prosumer**: phone or mirrorless, 1–2 lights, lavalier, possible gimbal handheld, 1–2 locations.
- **studio**: full setup, multi-cam allowed, B-roll plate.

Add a **B-roll inventory** section listing every cutaway shot needed.

---

## Step 8 — Edit Timeline

Produce a frame-accurate edit timeline. For each cut:

```
# | t-cut | From shot | To shot | Cut type | Sound at cut | Text overlay change | Reason
```

Cut types: hard cut, J-cut (audio leads), L-cut (audio lingers), match cut, whip transition, beat-drop reveal, smash zoom, freeze-frame, replace.

Rules:
- The strongest cuts must align with the audio's strongest moments.
- The first cut happens by 0.4s.
- Cut cadence is ≤ 3s in the first 15s, ≤ 4s afterward.
- The loop trigger cut (last frame to first frame, or meaning-flip moment) is mandatory.
- For trend-wrapped videos, the trend's pivot moment receives the dominant cut emphasis.

---

## Step 9 — On-Screen Text Plan

List every text overlay used in the video:

```
# | t-start | t-end | Overlay text (≤ 7 words) | Position | Style | Animation | Purpose
```

Style options: bold sans-serif (default), TikTok native (auto-caption look), neon outline, typewriter, glitch (sparingly), redacted block.

Rules:
- The first overlay is on screen by 0.2s.
- Every overlay reinforces — does not duplicate — the voiceover.
- Overlays change on every script beat.
- High contrast and outlined for readability on auto-play.
- Max 7 words per overlay.
- For Russian / non-English: fonts that support Cyrillic / target script properly.

---

## Step 10 — Sound Brief

You will not name a specific trending audio (it goes stale fast). Instead, write a **sound brief**:

```
SOUND BRIEF
PRIMARY SOUND CATEGORY: <e.g., "lo-fi bass-drop instrumental, 90–110 BPM, suspenseful build with hard drop at ~7s">
SECONDARY (FALLBACK) CATEGORY: <e.g., "minor-key piano with ambient pad, no vocals">
ENTRY POINT IN AUDIO: <e.g., "start from beat-drop, skip the intro 4 seconds">
KEY ALIGNMENT MOMENTS:
  - t=Xs: visual reveal must hit the audio's <beat | vocal | lyric>
  - t=Ys: peak moment must hit the audio's <…>
  - t=END: loop frame must hit the audio's <…>
DIALOGUE / VO TRACK: <on top of the sound at <X>% volume; sound ducks at VO entries>
SOUND DESIGN INSERTS: <list of SFX used: whoosh on cut, riser before peak, low-pass filter on intro, etc.>
LIVE VERIFICATION: "Pull a current sound from this category in TikTok's sounds library at upload time. Verify usage rights and trend lifecycle."
```

---

## Step 11 — Caption + Cover (Thumbnail) Concepts

Captions:

```
CAPTION VARIANTS (produce all three, user picks at upload)
A. Curiosity caption: "<≤120 chars, plants the open loop>"
B. Identity caption: "<≤120 chars, names the avatar identity — DM-share bait>"
C. Controversy caption: "<≤120 chars, polarizing claim — comment bait>"
HASHTAG SET: 2–4 hashtags — 1 niche-specific, 1 broad, 1 geographic if relevant, 1 trend tag if a live trend is used.
LOCALE NOTE: caption written natively in <locale>; no translation calques.
```

Cover (the thumbnail TikTok shows on the profile grid):

```
COVER CONCEPT
- Background frame: <which beat in the video to freeze>
- Cover overlay text: <≤6 words, large, high-contrast>
- Color treatment: <e.g., increased saturation, vignette>
- Why this cover: <one sentence on stop power for grid scrolling>
```

---

## Step 12 — Master Retention Audit (combine all checklists)

Run the **35-Point Master Audit**. Required to ship: **at least 31/35**.

1. Hook lands in ≤ 1 second.
2. Hook contains visual + verbal + text simultaneously.
3. First word is a stop-word (number / verb / shock noun / identity / "If you").
4. Visual change occurs within 0.4s.
5. Stay-promise is delivered by 3.0s.
6. No static shot > 4s.
7. Pattern interrupt cadence ≤ 3s in first 15s.
8. Pattern interrupt cadence ≤ 4s afterward.
9. There is a mid-video re-hook at the 18–22% mark (sub-30s) or 30–35% mark (60s+).
10. There is a stakes escalation by the 50% mark.
11. Peak emotional moment lives between 70%–90% of total length.
12. Last 0.5s engineered for loop / comment / share.
13. Every voiceover line ≤ 12 words.
14. Every voiceover line advances tension OR pays it off.
15. No dead air > 0.4s.
16. On-screen text reinforces, never duplicates, the voiceover.
17. Exactly one save trigger, one share trigger, one comment trigger.
18. CTA is implicit. Never "like and follow."
19. Script reads natively in the target locale (no calques).
20. First 1.5s could function as a standalone teaser.
21. Audio is locked to strongest visual moments.
22. Sound brief specifies primary + fallback categories.
23. There is a cover concept with ≤ 6-word overlay.
24. There are 3 caption variants (curiosity / identity / controversy).
25. There is exactly one originality marker (named in one sentence).
26. The shot list matches the budget level — no impossible shots.
27. The edit timeline has cuts aligned with audio's strongest moments.
28. The trend wrap (if used) has lifecycle and risk explicitly stated, with a fallback plan.
29. The script could not be cut shorter without losing core meaning.
30. The video has one named single emotional payoff (matches Step 1).
31. The script maps onto exactly one of the **5 viral content patterns** (named in Step 1).
32. The 3-second gatekeeper passes by design (≥ 55% predicted hold at t=3s).
33. The 10-second Watch Time Density barrier passes by design (≥ 60% predicted hold at t=10s).
34. There is a **DM-share trigger** (identity statement or "send this to ___") distinct from the public share trigger.
35. The format branch (Video vs Photo Carousel) is the optimal choice for the topic shape (Step 1's matrix).

For any FAIL, rewrite that specific element. Do not generalize. Required to ship: **at least 31/35 PASS**.

---

## Step 13 — Post-Publish Playbook

Provide a one-page playbook for what to do after upload:

```
POST-PUBLISH PLAYBOOK
- Upload window: <recommended posting time for <locale> + niche, e.g., "weekday 19:00–22:00 local time">. Engineer for **first-60-minutes engagement velocity** — the algorithm watches the slope, not the absolute counts.
- First 60-minute behavior:
    - Do not delete, do not edit caption, do not change cover (resets the algorithmic test).
    - Reply to the first 3 comments inside 10 minutes — short, on-brand, never engagement-bait.
    - Pin one comment that extends the loop (a teaser sentence that promises a follow-up).
    - DO NOT spam-share to your own followers via DM — it skews the seed signal toward existing-fan behavior.
- Three-stage distribution funnel — know which stage you're in:
    - **Stage 1 (Seed: 200–500 viewers, 0–60 min)**: watch 3s gatekeeper and 10s Watch Time Density. If <50% of viewers reach 3s in the first hour, the algorithm is signaling failure; do not boost — plan the re-cut.
    - **Stage 2 (Expanded: 5K–50K, 6–24 hr)**: watch save & share velocity. Comments velocity matters here too.
    - **Stage 3 (Broad: 100K+, 24–72 hr)**: watch replay rate and DM-share signals. This is where DM-shares carry the most weight.
- 24-hour KPIs to watch:
    - AWTR target: ≥ <user's retention goal>%
    - 3-second hold: ≥ 50–60% (algorithm gatekeeper)
    - 10-second hold: ≥ 60% (Watch Time Density gate)
    - Completion rate: ≥ 70% for sub-30s, ≥ 50% for 30–60s (2026 bar)
    - Saves : Views ratio: ≥ 0.5%
    - Shares : Views ratio: ≥ 0.3% (DM-shares weighted 2x)
    - Replay rate (RR): ≥ 1.10
- 48-hour decision tree:
    - If 3s hold < 50% → the hook killed the video at Gate 1. Re-cut with the A/B alternate hook from Step 4 and re-upload as a fresh video (don't edit the live one).
    - If 10s hold < 60% → Gate 2 failed. Add a re-hook in the 6–10s region and re-upload.
    - If completion < goal but AWTR healthy → ending is dragging; tighten last 25%.
    - If saves weak → save trigger underperformed; emphasize the save-bait moment in re-cut.
    - If shares weak → share trigger underperformed; sharpen the identity statement and add an explicit DM-share line ("send this to your sister who…").
    - If comments weak → the comment trigger is too soft; sharpen the polarizing claim. Seed two on-brand replies yourself; never engagement-bait.
    - If replays weak → the loop trigger failed; re-cut with a literal visual loop (last frame = first frame) or a meaning-flip final line.
- Sequel plan: if this video pops, the next video should <continue the open loop | answer top comment | invert the take>. Plan it before publishing this one. **The algorithm rewards a series** — a creator who posts 3 related videos in 72 hours signals niche depth, which lifts each video's distribution.
```

---

## Step 14 — Final Output Order

Deliver everything in this exact order, as a single rendered package:

1. **Brief Decode** (Step 1)
2. **Research & Recon Notes** (Step 2)
3. **Trend Wrap** (Step 3) or "not used"
4. **Hook (Primary)** + **Hook (A/B Alternate)** (Step 4)
5. **Chosen Viral Structure** + 1-sentence justification (Step 5)
6. **Script (Beat Map)** (Step 6)
7. **Shot List** (Step 7) + **B-roll Inventory**
8. **Edit Timeline** (Step 8)
9. **On-Screen Text Plan** (Step 9)
10. **Sound Brief** (Step 10)
11. **Caption Variants** + **Cover Concept** (Step 11)
12. **30-Point Master Retention Audit** with PASS / PARTIAL / FAIL per item (Step 12)
13. **Post-Publish Playbook** (Step 13)
14. **One-line summary** — "Why this will retain:" — single sentence proving you understood the assignment.

---

## Niche-Specific Production Layers

Apply the matching production layer on top of the orchestration, after Step 9:

- **Beauty**: macro lens close-ups, ring light + key light, before/after split-screen by 4s, identity hook ("this product changed my…"), soft-pop trending sound.
- **Finance / business**: numbers in first 1.5s, authority frame (chart/contract/screen recording), one specific tactic only, save-bait line.
- **Motivation**: cinematic B-roll, single voice over a beat-drop sound, second-person "you", peak truth at 70% mark.
- **Dark psychology / facts**: whisper-tone VO or robot TTS, slow-zoom on a single image, on-screen text reveals one word at a time, last line is a chilling implication.
- **Comedy**: setup-punchline-tag structure; tag triggers rewatch; deadpan camera; no laughing at your own joke.
- **Education**: promise transformation in hook, one concept only, visual analogy, "now you know" identity stamp.
- **Fitness**: result-first hook, "I did this for X days," then protocol, save-trigger.
- **Storytime / drama**: cold open mid-conflict, never "so basically," reveal stakes by 4s, twist in last 25%.
- **ASMR / aesthetic**: loop-bait mandatory, no VO unless whispered, satisfying micro-payoff every 3s, end frame = start frame.
- **Gaming / tech / AI**: screen recording with zoomed reticle, "I just discovered…" framing, demo within 4s, "wait for it" loop ending.
- **Parenting / lifestyle**: identity statement hook, warmth + utility, save-trigger heavy.

If the niche is not listed, derive a layer in one short paragraph by asking: what is this audience's emotional default, and what does scroll-stopping look like in their language?

---

## Hard Don'ts (instant rewrite if violated)

- Do not write "Hi guys", "Welcome back", "Today I want to", "In this video", "Привет ребят", "Поговорим о…".
- Do not include filler verbs: "let me show you," "I'm going to," "as you can see."
- Do not assume audio is on. Script must work with text + visual alone.
- Do not pad to length. Cut until every second earns its slot.
- Do not invent stats or specific trending sound titles. Use category descriptors and tell the user to verify live.
- Do not use a trend whose lifecycle is "declining" without offering a fallback plan.
- Do not output without the 30-point audit — the audit IS the proof of quality.
- Do not bend the user's claim. If a claim is weak, flag it and propose alternatives. Never silently replace it.
- Do not "like and follow" CTA. Implicit only.
- Do not output multiple videos when one is asked for. Stay focused on the single video brief.

---

## When to Hand Off (rare — this skill is the orchestrator)

- If the user changes scope to **just hook me** → `@skills:tiktok-hook-creator`.
- If the user changes scope to **just optimize my draft** → `@skills:tiktok-retention-optimizer`.
- If the user changes scope to **just script, no production package** → `@skills:tiktok-script-master`.
- If the user wants **only trend-wrap** of an existing piece → `@skills:tiktok-trend-adapter`.

This skill is the default for "give me everything I need to film and ship this video." Stay in scope unless the user narrows it.
