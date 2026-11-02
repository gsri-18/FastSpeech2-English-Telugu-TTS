# Gemini app (Argon High), listening check of the test sentences

All ten files from `Results/<Lang>/test_generated_audios/` were attached to one chat in the Gemini app, with the
model set to Argon High. The conversation had two steps.

1. `gemini_app_step1_prompt.txt` asked for a word-for-word transcript of each file without showing the intended
   text. The attachments reached the model without their file names, so it numbered its transcripts 1 to 10 in
   attachment order, and each was matched to its label from the transcript, since every test sentence is
   different.
2. `gemini_app_step2_message.txt` is the second message exactly as it was sent, with that matching at the top,
   followed by the intended text and the questions.

The replies are copied below without edits.

## Step 1 reply (blind transcripts)

1. తాడుపై అతని పట్టు చాలా బాగుంది
2. The orchestra's (unclear) conduct during the ceremony was praised
3. The meeting will start at 10 45 a m on 12 November 2026 [pause] so and all (unclear) should arrive on time
4. The ridem (unclear) of the orchestra changed suddenly
5. ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది
6. The researchers will conduct an experiment to evaluate the hypothesis
7. The antidisestablishmentarianism movement gained considerable momentum during the parliamentary debates
8. ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10 గంటల 45 నిమిషాలకు ప్రారంభమవుతుంది క ఆ (unclear) సమయానికి సిద్ధంగా ఉండండి
9. న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్
10. పులి అడవిలో నిశ్శబ్దంగా సంచ

## Step 2 reply (judgements)

E1

1. Intelligibility: 2
2. Naturalness: 1
3. Robotic buzz: strong
4. Words that sound wrong:
  ⚬ "antidisestablishmentarianism" sounds rushed, muffled, and compressed, starting as "an-tai-dis-establish-men-" before slurring the ending into "-tee-in-ism" / "-tair-ee-un-ism".
  ⚬ "gained" has a weak, swallowed initial consonant and sounds almost like "in" or "gain".
  ⚬ "parliamentary" is compressed and slurred into "par-luh-men-tree".
  ⚬ "proceedings" is not spoken at all; the recording says "debates" (sounding muffled like "di-baits") instead.
5. Specific question: "antidisestablishmentarianism" is only partially intelligible—you can catch "an-tai-dis-establish-" if you listen closely, but the rapid syllable compression and heavy vocoder distortion make the second half hard to parse without already knowing the word. "proceedings" cannot be judged for stretching because the word is completely absent from the audio, which ends with "debates" instead.
6. Anything else a listener would notice: The entire clip has a muffled, low-bandwidth, metallic vocoder quality with flat, rushed prosody and background hum from start to finish.

E2a

1. Intelligibility: 3
2. Naturalness: 1
3. Robotic buzz: strong
4. Words that sound wrong:
  ⚬ "orchestra's" blends with "The" and shifts prominence weirdly onto the middle syllable, sounding muffled like "wor-kus-truz" or "or-kus-truz".
  ⚬ "ceremony" is compressed in the middle, sounding close to "ser-muh-nee".
  ⚬ "by the audience" is completely missing—the recording stops right after "praised".
5. Specific question: "conduct" is stressed on the first syllable as CON-duct (with a full "kon" vowel, though the second syllable is not heavily reduced either). "orchestra" sounds very unnatural and garbled—the muffled vowels, heavy buzz, and awkward second-syllable weight make "orchestra's" easy to mishear on first listen.
6. Anything else a listener would notice: Aside from the missing final phrase ("by the audience"), the speech sounds heavily vocoded, nasal, and monotone throughout.

E2b

1. Intelligibility: 3
2. Naturalness: 1
3. Robotic buzz: strong
4. Words that sound wrong:
  ⚬ "will" has a weak final "l" and sounds muffled, almost like "wee" or "wi".
  ⚬ "experiment" is slightly compressed in the middle syllables ("ik-sper-ment" / "ik-spe-ri-ment").
  ⚬ "proposed method" is not spoken at all; the recording says "hypothesis" ("hai-po-thuh-sis") instead.
5. Specific question: "conduct" is stressed on the second syllable as con-DUCT, with a shorter, lighter first syllable leading into "-DUCT".
6. Anything else a listener would notice: Sibilants in "researchers" and "hypothesis" carry harsh metallic distortion, and the final words differ completely from the intended script.

E3

1. Intelligibility: 2
2. Naturalness: 1
3. Robotic buzz: strong
4. Words that sound wrong:
  ⚬ "rhyth m" is mispronounced with a long "ai" diphthong, sounding like "ride-em" or "rai-dem".
  ⚬ "orchestra" sounds muffled and buzzy ("or-kus-truh"), blending into "changed".
  ⚬ "unexpectedly" is not spoken; the recording actually says "suddenly" ("sud-uhn-lee").
5. Specific question: Instead of normal "rhythm" (/ˈrɪðəm/), "rhyth m" is heard continuously without a long gap as "rai-dem" / "ride-em"—the synthesiser reads "rhyth" with an "ai" vowel (like ride) and merges it straight into "em" for the standalone "m".
6. Anything else a listener would notice: Beyond the broken pronunciation of "rhyth m" and the substitution of "suddenly" for "unexpectedly", the clip shares the same harsh, raspy vocoder buzz as the other English recordings.

E4

1. Intelligibility: 2
2. Naturalness: 1
3. Robotic buzz: strong
4. Words that sound wrong:
  ⚬ "AM" is compressed and slurred right into "on", sounding like a muffled "em" or "aym".
  ⚬ "November" is muffled and compressed, sounding close to "noo-vuhm-bur".
  ⚬ "2026" has heavy distortion on the first "twenty", making the onset sound muddled before "twenty-six" clears up.
  ⚬ "andaroo" sounds anglicised and slurred into surrounding words, resembling "and all" or "an-duh-roo".
5. Specific question: The time is unclear—there is definitely no "p" sound for PM, only a brief muffled vowel-plus-"m" sound ("em" / "aym") squeezed between "45" and "on" that is too slurred to make out cleanly as "A.M.". "andaroo" is not intelligible without seeing the text; spoken with an English alveolar "d", reduced vowels, and heavy vocoder muffling, it sounds like garbled English words ("and all" / "an-duh-roo") rather than the Telugu word అందరూ. The pause after "2026" is too long—it feels like an unnatural full-second gap between two separate recordings rather than a normal comma break.
6. Anything else a listener would notice: Background static/hum remains audible even around the long mid-sentence pause, making the splice-like silence stand out sharply.

T1

1. Intelligibility: 4
2. Naturalness: 3
3. Robotic buzz: none
4. Words that sound wrong:
  ⚬ "విశ్లేషించారు" is abruptly cut off mid-word right after "విశ్" ("viś-"), dropping "-లేషించారు" completely.
5. Specific question: The first vowel of ఔషధాల is pronounced clearly and accurately as the diphthong "au" ("au-ṣa-dhā-la"), gliding cleanly from "a" to "u". The long opening word న్యూరోఫార్మకోలాజికల్ is completely clear and easy to follow, spoken crisply in Telugu phonology with slight rhythmic grouping ("న్యూరో-ఫార్మ-కోలాజికల్").
6. Anything else a listener would notice: Compared to the English set, the Telugu voice is dramatically cleaner and free of vocoder buzz, though పరిశోధనలో runs straight into నూతన with almost no pause and the recording terminates suddenly mid-syllable at the end.

T2a

1. Intelligibility: 4
2. Naturalness: 4
3. Robotic buzz: none
4. Words that sound wrong:
  ⚬ "తాడుపై అతని" has an awkward boundary shift where తాడు separates slightly from పై, and పై attaches to అతని so it sounds like "తాడు పయతని" ("tāḍu payatani").
  ⚬ "బలంగా ఉంది" is not spoken; the recording says "బాగుంది" ("bāgundi") instead, with a slightly clipped final "-ది".
5. Specific question: What is actually heard (తాడుపై అతని పట్టు చాలా బాగుంది — "His grip on the rope is very good") is natural, grammatical everyday Telugu. Individual words like పట్టు (with clear geminate ట్ట) and చాలా బాగుంది are pronounced accurately, with the only pronunciation flaw being the awkward sandhi-like running together of పై + అతని into పయతని.
6. Anything else a listener would notice: The voice timbre, pitch contour, and lively delivery sound smooth and human-like, despite substituting బాగుంది for the intended బలంగా ఉంది.

T2b

1. Intelligibility: 4
2. Naturalness: 3
3. Robotic buzz: none
4. Words that sound wrong:
  ⚬ "ఆమె" has a faint breathy/aspirated artefact right at the onset.
  ⚬ "వేడుకకు వచ్చింది" is not what is heard; the recording says "వేడుకలో" ("vēḍukalō") and then cuts off abruptly on the very first consonant onset of the next word.
5. Specific question: Both the intended sentence and the spoken portion (ఆమె పట్టుచీరను ధరించి వేడుకలో...) are natural formal/narrative Telugu. None of the completed words (ఆమె, పట్టుచీరను, ధరించి, వేడుకలో) are mispronounced—the retroflex geminate in పట్టు ("silk") and vowel lengths in చీరను are crisp—but the sentence is left incomplete due to the sudden cutoff after వేడుకలో.
6. Anything else a listener would notice: Just like T1 and T3, the audio truncates abruptly before finishing the final verb, suggesting a hard duration cutoff during generation.

T3

1. Intelligibility: 4
2. Naturalness: 3
3. Robotic buzz: none
4. Words that sound wrong:
  ⚬ "నిశ్శబ్దంగా" has its sibilant geminate articulated slightly closer to dental/alveolar "నిస్సబ్దంగా", though it still sounds natural in spoken Telugu.
  ⚬ "నడిచింది" is not spoken; instead, the speaker starts saying "సంచ—" ("san̄ca-", as in సంచరించింది) and cuts off abruptly mid-word.
5. Specific question: అడ విలో is heard smoothly as the single word అడవిలో ("aḍavilō"), with no pause, glitch, or internal break inside it.
6. Anything else a listener would notice: The pacing and tone are clean and natural right up until the recording cuts off mid-word on "సంచ—".

T4

1. Intelligibility: 4
2. Naturalness: 3
3. Robotic buzz: none
4. Words that sound wrong:
  ⚬ Right at the comma after "ప్రారంభమవుతుంది", there is an extraneous stutter/vocal glitch sounding like "క-ఆ" ("ka-aa") before "సమయానికి" begins.
  ⚬ "ఉంచండి" ("keep ready") is spoken instead as "ఉండండి" ("uṇḍaṇḍi", "be ready").
5. Specific question: Yes—all the numbers are expanded into natural Telugu words and spoken clearly and accurately: 2026 as రెండు వేల ఇరవై ఆరు, 12వ with proper ordinal inflection as పన్నెండవ, and 10:45 గంటలకు expanded idiomatically as పది గంటల నలభై ఐదు నిమిషాలకు ("at 10 hours 45 minutes").
6. Anything else a listener would notice: The stray "క-ఆ" syllable at the comma is very jarring in an otherwise fluent announcement, and ఉండండి at the end actually fits the sentence meaning ("be ready on time") more naturally than the intended ఉంచండి.
