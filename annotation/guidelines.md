# TerraLens labelling guideline (v1)

**Claim:** "Climate change and global warming is a serious concern."

For each comment, decide what the **writer** believes about this claim. Label the comment, not the
video, and not the topic.

| Label | Use when the writer... |
|---|---|
| **Favour** | agrees climate change / global warming is real, serious, harmful, urgent, or worth acting on; mocks or criticises deniers; worries about heat, floods, melting glaciers *as signs of* climate change; calls for action (plant trees, cut pollution) *because of* climate change. |
| **Against** | denies it, calls it a hoax, propaganda, scam, Western agenda or exaggeration; says it is natural / normal / always happened so not a concern; says other issues matter and climate does not; mocks people who worry about it. |
| **None** | takes no position on the claim: off-topic, about the video quality/anchor/channel, greetings, spam, pure questions, neutral facts with no judgement, politics or blame without a view on seriousness, or genuinely unclear. |

## Decision rules

1. **Target, not topic.** Mentioning climate or weather is not enough. "Aaj bahut garmi hai" (it's very
   hot today) with no link to climate change is **None**.
2. **Intended meaning wins.** Label sarcasm by what it means. "Wah, April mein 48 degree, sab normal hai"
   ("Wow, 48°C in April, all normal") mocks denial → **Favour**.
3. **Negation.** "Climate change jhooth nahi hai" (climate change is not a lie) → **Favour**.
   "Ye koi crisis nahi hai" (this is no crisis) → **Against**.
4. **Implicit stance counts** if a reasonable reader would infer it. "Har saal garmi badh rahi hai, bachchon
   ka kya hoga?" (heat rises every year, what will happen to our children?) → **Favour**.
5. **Blame without judgement is None.** "Sab China ki galti hai" (it's all China's fault) → **None**, unless
   it also says the problem is real/serious ("China ki wajah se dharti garam ho rahi hai, ye khatarnaak hai"
   → **Favour**).
6. **Questions.** A genuine question with no lean → **None**. A rhetorical question with a clear lean →
   label the lean ("Global warming? Toh phir itni thand kyun hai?" → **Against**).
7. **Mixed or contradictory** comments: label the dominant final position; if truly balanced → **None**.
8. **Praise for the video** alone ("bahut accha video sir") → **None**. Praise plus agreement with the
   message ("sahi kaha sir, ab bhi nahi jaage toh der ho jayegi") → **Favour**.
9. **Language does not matter.** Native script, romanised and code-mixed comments follow the same rules.
10. If you are below about 60% confident, choose **None** and set `confidence: low`.

## Worked examples (written by us for the guideline, not from the dataset)

| # | Comment | English gloss | Label | Why |
|---|---|---|---|---|
| 1 | हर साल गर्मी बढ़ रही है, अब कदम उठाना ज़रूरी है | Heat rises every year; we must act now. | Favour | explicit concern + action |
| 2 | ये सब वेस्टर्न देशों का प्रोपेगेंडा है, मौसम तो हमेशा बदलता है | It's all Western propaganda; weather always changes. | Against | hoax + natural variation |
| 3 | Video ka audio thoda low hai, please fix karo | The audio is low, please fix it. | None | about the video |
| 4 | climate change jhooth nahi hai bhai, glacier pighal rahe hain | Climate change is not a lie, glaciers are melting. | Favour | negated denial |
| 5 | Wah, April mein 48 degree, sab normal hai 🙂 | Wow, 48°C in April, all normal. | Favour | sarcasm against denial |
| 6 | global warming ek scam hai, carbon tax ke liye bana hai | Global warming is a scam made for carbon tax. | Against | scam claim |
| 7 | Sab sarkar ki galti hai | It's all the government's fault. | None | blame, no view on seriousness |
| 8 | Agar global warming hai toh Delhi mein itni thand kyun? | If there's global warming, why so cold in Delhi? | Against | rhetorical denial |
| 9 | ক্লাইমেট চেঞ্জ সব বানানো গল্প | Climate change is all a made-up story. | Against | denial |
| 10 | প্রতি বছর বন্যা বাড়ছে, এখনই কিছু না করলে বিপদ | Floods grow every year; danger if we don't act now. | Favour | concern + action |
| 11 | দাদা খুব সুন্দর ভিডিও | Brother, very nice video. | None | praise only |
| 12 | global warming ki baat sab karte hain par koi kuch nahi karta | Everyone talks about global warming but no one acts. | Favour | implies it needs action |
| 13 | Isse zyada important hai berozgari, ye climate wagairah baad mein | Unemployment matters more; this climate stuff later. | Against | dismisses as not a concern |
| 14 | Climate change kya hota hai exactly? | What exactly is climate change? | None | genuine question |
| 15 | এটা প্রাকৃতিক চক্র, ভয় পাওয়ার কিছু নেই | It's a natural cycle, nothing to fear. | Against | natural + not a concern |
| 16 | গরম তো সবসময়ই ছিল | It was always hot. | Against | implies nothing unusual |
| 17 | Ped lagao, dharti bachao 🌳 | Plant trees, save the earth. | Favour | action slogan implying concern |
| 18 | Subscribe karo mere channel ko | Subscribe to my channel. | None | spam |

## Output format for LLM annotators

One JSON object per comment:

```json
{"id": "yt_abc123_0042", "label": "Favour", "confidence": "high", "gloss": "English translation", "reason": "one short sentence"}
```

`label` ∈ {Favour, Against, None}; `confidence` ∈ {high, medium, low}.
Also tag phenomena for the robustness analysis: `"tags": ["sarcasm", "negation", "implicit", "code_mixed", "romanised", "question"]` (any that apply, else `[]`).
