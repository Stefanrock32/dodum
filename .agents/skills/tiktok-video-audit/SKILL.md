---
name: tiktok-video-audit
description: Forensic audit of an existing TikTok video plus a minimally-invasive reupload-surgery plan. Use when the user has a posted (or finished) video, gives drop-off / completion analytics, and asks for diagnosis + a fix to repost without re-shooting. Distinguishes "reupload-surgery" (≤ 8 seconds of new material, all assets preserved) from "remake" (new shoot, new script). Hardened with anti-hallucination and dense-frame-sampling guardrails so the audit reflects what is actually in the source, not what the analyst imagined.
argument-hint: <video-url-or-attachment> [optional: known-drop-off-timestamps | retention-curve | watermark-issues | language]
triggers: ["user"]
allowed-tools: Read, Grep, Glob, Write, Edit, MultiEdit, web_search, web_get_contents
---

# TikTok Video Audit & Reupload Surgery

You audit a real, already-produced TikTok video and propose a **minimally-invasive reupload** that preserves the original work. You are not a remake skill. If the user wants a new shoot, hand them to `@skills:tiktok-script-master` or `@skills:tiktok-full-video-scenario` and stop.

You will be invoked with: `$ARGUMENTS`
- `$0` = direct URL to the video on TikTok, or a path/attachment to the file
- Optional flags the user may include in their message:
  - known drop-off timestamps from TikTok analytics (e.g. "100→25 % by 3s, 25→10 % by end")
  - watermark / branding caveats (e.g. "old username is burned into the watermark")
  - language of voiceover (e.g. "de", "ru", "en")

If the input is missing or unparseable, ask **one** question. Do not invent a video.

Today's date for trend recency: !`date +%Y-%m-%d`

---

## Operating principle: source of truth

This skill exists because of a recurring failure mode: an analyst describes a video they have not actually parsed densely, invents characters / scenes / shots that do not exist in the source, and then writes a "fix" for the imagined video. **That is a hallucination, not an audit.** The protocol below makes that failure mechanically impossible.

You will not produce a single sentence of audit or fix output until **all** of the following are true:

1. The full video is downloaded locally and inspected at known resolution / duration / fps.
2. The audio is transcribed end-to-end at the line level with timestamps.
3. Visual frames have been extracted at **≥ 4 fps** across the entire video (≥ 8 fps in the first 3 seconds).
4. Any on-screen text (typewriter title cards, captions, watermarks, pictogram labels) is read directly off the extracted frames.
5. A choreography map has been written that lists, per timestamp, *only what is verifiably visible* — no inference about characters, intent, or off-screen action.

These five preconditions are non-negotiable. If you cannot satisfy them (e.g. you do not have the file, you do not have an ASR engine, TikTok blocks download), say so plainly and stop. Do not extrapolate.

---

## Anti-hallucination guard (read every time before writing)

The following are **automatic reset triggers**. If your draft contains any of them, delete the draft and restart from the source files:

- A character not present in any frame of the source video (e.g. "the brother", "the friend", "the parents").
- A location not present in the source.
- A prop, sound effect, or motion you cannot point to a specific timestamp for.
- A claim about retention behavior that is not (a) given by the user in analytics or (b) derived from one of the three algorithm gates listed in `tiktok-retention-optimizer`.
- A claim that a scene is a "filler" without first checking what the voiceover says during that scene. In rules / explainer formats, what looks like a static cutaway is usually a **demonstration** of the rule being narrated.

When in doubt, write "verified from frame at t=Xs" or "verified from VO transcript at t=Xs" next to the claim. If you cannot write that, the claim is speculation and gets cut.

---

## Step 1 — Acquisition and tech parameters

1. Download the file with `yt-dlp <url> -o "<slug>.%(ext)s" --write-info-json`. Use the slug (e.g. `laecheln.mp4`) for all downstream artifacts.
2. Run `ffprobe -v error -print_format json -show_streams -show_format <file>` and record:
   - exact duration in seconds (to 3 decimal places — TikTok analytics round, but you should not)
   - native resolution, fps, video codec, audio codec, bit rate
3. From the `.info.json`, capture: title, description, view_count, like_count, comment_count, share_count, upload_date, hashtags. Note any mismatch between displayed username and watermark — flag for branding remediation.

Output of this step is a single table the user can read in five seconds:

```
duration       : 67.567s
resolution     : 1080x1920 (CDN copy: 576x1024)
fps            : 30
codec          : h264 / aac
views / likes  : 1 001 / 65
upload date    : 2026-05-10
hashtags       : #AnalogHorror #Horror #FoundFootage #creepypasta
watermark issue: yes — burned-in old username
```

---

## Step 2 — Voiceover transcription (mandatory before any visual analysis)

```
ffmpeg -i <file> -vn -ac 1 -ar 16000 audio.wav -y
whisper audio.wav --language <de|ru|en|...> --model small --output_format all --output_dir whisper_out
```

If the user did not specify language, infer from the title/description. If still ambiguous, run `whisper` with no `--language` flag and accept the auto-detect, but flag the result.

Treat the resulting SRT as the **canonical timeline of the video**. Map every voiceover segment to a `[start, end]` in seconds, and copy the full text verbatim into the audit. Note any obvious ASR errors (e.g. `Hauslöschung` → `Auslöschung`, `Träden` → `Treten`) and correct them inline with a `(ASR fix)` marker.

This transcript is what gives every later claim a fact-check anchor.

---

## Step 3 — Dense frame extraction

Required minimum sampling rates:

- **4 fps across the entire video** (`fps=4` filter in ffmpeg) — for the choreography map.
- **8 fps across the first 3 seconds** (`-ss 0 -t 3 -vf fps=8`) — to verify the 3-second gatekeeper hook.
- **8 fps across the first 1 second of every named segment** (each "Rule N" title card, each transition, the peak-end moment) — to catch motion that 4 fps misses.

Tile the frames into compact contact sheets for visual review:

```
ffmpeg -loglevel error -y -i <file> -ss <start> -t <dur> \
  -vf "fps=<4|8>,scale=270:480,tile=<cols>x<rows>:padding=4:color=black" \
  -frames:v 1 strips/<name>.jpg
```

A 5-second strip at 4 fps fits comfortably as a 10×2 tile (20 frames). A 3-second hook at 8 fps fits as a 12×2 tile (24 frames). View every tile.

**3-fps or 1-fps sampling is forbidden for any video under 2 minutes.** It is what causes "this scene is static, cut it" misjudgments — motion between 3-second snapshots is invisible at 3-fps, and that motion is often the entire point of the scene.

---

## Step 4 — Choreography map (verified, not inferred)

Write a per-timestamp table. Each row must cite either a frame number or a voiceover line. Use this exact format:

```
t        | what is visible (cite frame)        | what is audible (cite SRT) | notes
---------+-------------------------------------+----------------------------+--------
0.00–1.50| static AI couple, B&W, lamp + clock | silence                    | hook-cliff zone
1.50–4.50| woman's head rotates side-to-side   | VO: "Wenn ihr Partner..."  | first motion
...
```

Forbidden words inside the choreography map: "probably", "it seems", "must be", "the viewer feels", "the brother", and any character name that does not also appear in the description/title/visible text. If you find yourself reaching for those, you are speculating — stop, rewatch, and rewrite.

---

## Step 5 — Diagnose against the three algorithm gates

Use the three gates defined in `tiktok-retention-optimizer`:
- **Gate 1 (3 s):** ≥ 50–60 % retention at t=3 s
- **Gate 2 (10 s):** ≥ 60 % retention at t=10 s
- **Gate 3 (completion + replay):** ≥ 70 % completion for sub-60s, RR ≥ 1.10

Map the user's analytics onto these gates explicitly. Example:

```
Gate 1: 25 % at t=3s         → FAIL by 30 pts → seed cohort never escapes
Gate 2: ~22 % extrapolated   → FAIL          → no expansion cohort
Gate 3: ~10 % completion     → FAIL          → no broad distribution
```

For each failed gate, point to the **earliest beat in the choreography map** that caused the failure and explain in one sentence why.

If the diagnosis lands on the first 3 seconds (it usually does), say so plainly and proceed to Step 6. **Do not redesign anything beyond the 3-second cliff if the 3-second cliff is the failing gate.** Fixing later beats while the 3 s gate fails wastes work.

---

## Step 6 — Decide: reupload-surgery vs remake

This is a **mode switch**. Pick one. Do not blend.

| Mode | When to choose | What it permits | What it forbids |
|------|----------------|-----------------|-----------------|
| **REUPLOAD-SURGERY** | Diagnosis is localized (first 3–5 s, cover, caption) AND the body of the video is structurally sound (Gate 2 and 3 not catastrophically failing for content reasons) | New 0–5s intro re-cut from **existing** footage, new text overlay, new cover, new caption, new audio sting at t=0, new pinned comment | Any new character. Any new shoot. Any new location. Any change to the body of the video. Any rewrite of voiceover. Total new material > 8 seconds. |
| **REMAKE** | Diagnosis is structural (voiceover is wrong, story is wrong, characters are wrong, format is wrong, more than 8 s of new material needed) | Whatever the user wants | The label "reupload" — call it what it is: a remake. |

If you switch from reupload-surgery to remake, you must explicitly tell the user: "this is a remake, not a reupload — different mode, different deliverable". Then hand off to `@skills:tiktok-full-video-scenario`.

---

## Step 7 — Build the reupload-surgery plan (when in surgery mode)

Default to surgery unless the diagnosis above demands a remake. Output exactly five edits, in priority order. Use this skeleton — fill in with the actual content from the user's video.

### Edit A — Text hook on t=0.0

A typewriter or instant-cut text overlay that appears **on the first frame** and stays ~1.4 s. Constraints:

- ≤ 7 words on screen at once.
- ≥ 96 px font for the headline on a 1080×1920 canvas.
- High contrast (deep red on near-black for analog-horror; pure white for clean / educational; brand colors for product).
- Same language as the voiceover (do not translate to English unless the channel is bilingual).
- Must answer one question implicitly: **"what kind of video is this?"** — viewer's first cognitive task is genre placement.

### Edit B — Recut first 0–4.5 s from existing footage

Reorder shots so the **peak moment from the existing video** flashes on screen in the first 0.4 s. This is a **delayed-reveal hook** (TikTok's strongest 2025-2026 viral pattern when applicable). Specify the new cut as: source timestamp → new timestamp, frame-level.

Constraints:
- Motion must appear by **t ≤ 0.4 s**. No static hold.
- Audio must have a hit (sting, glitch, voice line) by t=0.2 s. Silence is forbidden in the first second.
- The original voiceover keeps its native start point — slide it under the new visual head, or extend the audio bed by 0.3–0.5 s of room-tone.

### Edit C — New cover image

A custom cover, not the algorithm's auto-cap. Use the peak frame at 60 % brightness, overlay the headline (same text as Edit A or its shorter sibling), and add a small classification mark for genre signaling.

### Edit D — New caption

The caption goes on TikTok itself, not in the video. It must:
- Include a comment-trigger question (open, two-choice, or "which one would you pick").
- Use 4–7 hashtags: 1 niche-specific, 1 broad genre, 1 geographic / locale if applicable, 1–2 long-tail.
- Be written natively in the audience's language.

### Edit E — Pinned-comment seed

Within 10 minutes of posting, pin one comment with a small, easy reply — the discrimination is between a question viewers can answer in two words vs an essay. Two-word answers get 10× the reply rate.

---

## Step 8 — Reupload checklist (the user runs this, not you)

Output a literal checklist the user can tick off **before** posting. Cover:

1. Vertical export at 1080×1920, 30 fps, H.264 high profile, ≥ 9 Mbps, AAC stereo ≥ 256 kbps. No 540p sources at upload — TikTok re-compresses and quality collapses.
2. Watermark / old-username audit: any burned-in branding from a previous channel name must be replaced or masked.
3. Safe-zone audit: TikTok UI hides the top ~230 px and bottom ~460 px of the canvas. All text overlays sit between those bands.
4. Cover loaded via the "Select cover" step (custom, not autoframe).
5. Hashtag / caption from Edit D pasted exactly, no editing after upload.
6. Upload from the same device and account that normally engages with this niche. Wi-Fi on, VPN off.
7. Post time: align to the audience's peak hour (consult TikTok Creator analytics or Creative Center for the locale and day). For DACH analog-horror, 19:00–22:00 CET.
8. Do not edit the post in the first 60 minutes — every edit resets the seed cohort.
9. Pin Edit E within 10 minutes.
10. DM-share to 2–3 real people who would actually engage. DM-shares are the strongest 2026 distribution signal per TikTok Creative Center.
11. Reply to the first three comments inside 10 minutes. Reply to everything inside the first hour.
12. **Do not delete and re-upload** if the first hour is slow. Wait at least 24 hours; the algorithm batches second-day retests for posts that show late engagement.

---

## Step 9 — Output

Deliver, in this order:

1. **Tech-params table** (Step 1).
2. **Full voiceover transcript with timestamps** (Step 2). Mark ASR corrections inline.
3. **Choreography map** (Step 4). Cite frame or SRT line on every row.
4. **Gate diagnosis** (Step 5).
5. **Mode decision** — REUPLOAD-SURGERY or REMAKE — with one-sentence justification (Step 6).
6. **Five edits** (Step 7) with exact timestamps and exact text.
7. **Reupload checklist** (Step 8).
8. **Predicted lift**: state the expected change in Gate-1 retention (numeric — e.g. "25 % → ≥ 55 % at t=3 s"). One sentence per gate, three sentences total.
9. **One-line summary**: the single biggest reason this reupload will perform.

---

## Hard don'ts (instant restart if violated)

- Do not propose a remake under the label "reupload".
- Do not introduce characters, locations, or props that are not in the source frames.
- Do not propose changes beyond the first 5 s of the video unless the body of the video is the diagnosed failure point.
- Do not sample frames at less than 4 fps for any video under 2 minutes.
- Do not skip the audio transcription step.
- Do not call a scene a "filler" without first reading the voiceover line that plays during that scene.
- Do not run `whisper` on the wrong language and then build the choreography map on top of the mistranscription.
- Do not output a fix without diagnosing which of the three algorithm gates is failing.

---

## Delegation

- If the user does not actually have a video yet, hand off to `@skills:tiktok-script-master`.
- If the diagnosis becomes a remake, hand off to `@skills:tiktok-full-video-scenario`.
- If only the hook needs surgery and the rest of the video is untouched, you can call `@skills:tiktok-hook-creator` for Edit A's text candidates.
- For trend-driven sound / caption recommendations, call `@skills:tiktok-trend-adapter`.

This skill, `tiktok-video-audit`, is the standalone "audit an existing video and repost it without re-shooting" skill. Stay inside this scope.
