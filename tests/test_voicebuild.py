"""Tests for voicebuild's stdlib parts: subtitle parsing and cleaning, stream choice, sync fitting.

Run with: python3 -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voicebuild import media, subtitles, sync  # noqa: E402

SRT = """\ufeff1
00:01:02,500 --> 00:01:04,000
HAL: Good afternoon, gentlemen.

2
00:01:05.250 --> 00:01:07,750
-How are you?
-Fine.

00:01:08,000 --> 00:01:09,000
<i>[music playing]</i> Hello there.
"""

VTT = """WEBVTT

00:00:01.000 --> 00:00:02.500 align:start
First line

1:02:03.400 --> 1:02:05.000
Second line
"""

ASS = """[Script Info]
Title: test

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:10.00,0:00:12.50,Default,Hal,0,0,0,,I'm sorry, Dave.\\NI'm afraid I can't do that.
Comment: 0,0:00:13.00,0:00:14.00,Default,,0,0,0,,not shown
Dialogue: 0,0:00:15.00,0:00:16.00,Default,,0,0,0,,{\\i1}Plain{\\i0} line
"""


class ParseTest(unittest.TestCase):
    def test_srt(self):
        rows = subtitles.parse_srt(SRT)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][:2], (62.5, 64.0))
        self.assertEqual(rows[1][0], 65.25)
        self.assertEqual(rows[2][2], "<i>[music playing]</i> Hello there.")

    def test_vtt(self):
        rows = subtitles.parse_srt(VTT)
        self.assertEqual([r[0] for r in rows], [1.0, 3723.4])

    def test_ass_names_become_labels(self):
        rows = subtitles.parse_ass(ASS)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0][2].startswith("HAL: I'm sorry, Dave.\nI'm afraid"))
        self.assertEqual(rows[1][2], "Plain line")

    def test_encodings(self):
        tmp = Path(__file__).with_name("_cp1252.srt")
        tmp.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nCaf\xe9\n".encode("cp1252"))
        try:
            self.assertEqual(subtitles.read_file(tmp)[0][2], "Caf\xe9")
        finally:
            tmp.unlink()


class CueTest(unittest.TestCase):
    def test_labels_and_dashes_split_speakers(self):
        parts = subtitles.split_speakers("HAL: Good evening, Dave.\nBOWMAN: How are you, Hal?")
        self.assertEqual([p[0] for p in parts], ["HAL", "BOWMAN"])
        parts = subtitles.split_speakers("-Who's taking care of you?\n-Rachel.")
        self.assertEqual([p[2] for p in parts], ["Who's taking care of you?", "Rachel."])

    def test_label_tags(self):
        spk, tags, text = subtitles.split_speakers("HAL SINGS:\nDaisy")[0]
        self.assertEqual((spk, tags, text), ("HAL", ["sings"], "Daisy"))
        spk, tags, _ = subtitles.split_speakers("BOWMAN (ON RADIO): Hello")[0]
        self.assertEqual((spk, tags), ("BOWMAN", ["on radio"]))

    def test_sound_cues_removed(self):
        self.assertEqual(subtitles.clean("[ALARM BLARING] Dave, stop. \u266a"), "Dave, stop.")

    def test_ocr_fixes(self):
        self.assertEqual(subtitles.fix_ocr("| think you missed it."), "I think you missed it.")
        self.assertEqual(subtitles.fix_ocr("/'m sorry, Frank."), "I'm sorry, Frank.")
        self.assertEqual(subtitles.fix_ocr("ll meet you there."), "I'll meet you there.")
        self.assertEqual(subtitles.fix_ocr("The radio ts still dead."), "The radio is still dead.")
        self.assertEqual(subtitles.fix_ocr("When we found Itt..."), "When we found It...")

    def test_two_speaker_subtitle_time_is_shared(self):
        out = subtitles.cues([(10.0, 14.0, "HAL: Good evening, Dave.\nBOWMAN: Hi.")])
        self.assertEqual(out[0].start, 10.0)
        self.assertAlmostEqual(out[1].end, 14.0)
        self.assertGreater(out[0].end - out[0].start, out[1].end - out[1].start)


class StreamTest(unittest.TestCase):
    def streams(self):
        S = media.Stream
        return [
            S(1, "audio", "aac", "eng", "", 6, "5.1", default=True),
            S(2, "audio", "aac", "eng", "Commentary", 2, "stereo"),
            S(3, "subtitle", "dvd_subtitle", "eng", ""),
            S(4, "subtitle", "subrip", "eng", "Forced", forced=True),
            S(5, "subtitle", "subrip", "eng", "English SDH", hearing_impaired=True),
            S(6, "subtitle", "subrip", "fre", ""),
        ]

    def test_picks_dialogue_track_and_full_english_text_subtitles(self):
        self.assertEqual(media.pick_audio(self.streams()).index, 1)
        self.assertEqual(media.pick_subtitles(self.streams()).index, 5)

    def test_bitmap_used_when_no_text(self):
        only_bitmap = [s for s in self.streams() if s.index != 5]
        self.assertEqual(media.pick_subtitles(only_bitmap).index, 3)

    def test_centre_channel_for_surround(self):
        S = media.Stream
        self.assertEqual(
            media.dialogue_filter(S(0, "audio", "ac3", "eng", "", 6, "5.1(side)")), "pan=mono|c0=FC"
        )
        self.assertIn("0.5*c0", media.dialogue_filter(S(0, "audio", "aac", "eng", "", 2, "stereo")))


def film(offset: float, drift: float, jump_at: float = 1e9, jump: float = 0.0, slip: bool = False):
    """Returns subtitle cues and recognised words for a synthetic film with known timing errors."""
    import random

    rnd = random.Random(1)
    vocab = (
        "open the pod bay doors hal i am sorry dave afraid cannot do that mission important".split()
    )
    cues, words = [], []
    for n in range(60):
        t = 30.0 + n * 40.0
        said = [rnd.choice(vocab) for _ in range(rnd.randint(3, 7))]
        text = " ".join(said).capitalize() + "."
        if slip and n % 5 == 0:
            text = text.replace("i ", "| ", 1)
        cues.append(subtitles.Cue(start=t, end=t + 0.4 * len(said), text=text, cue=n))
        real = t + offset + drift * t + (jump if t > jump_at else 0.0)
        for k, w in enumerate(said):
            words.append(
                {"start": real + 0.4 * k, "end": real + 0.4 * k + 0.3, "word": w, "prob": 0.9}
            )
    return cues, words


class SyncTest(unittest.TestCase):
    def test_offset_and_drift_are_recovered(self):
        cues, words = film(offset=2.5, drift=0.001)
        lines, report = sync.align(cues, words)
        self.assertAlmostEqual(report["offset_s"], 2.5, delta=0.05)
        self.assertAlmostEqual(report["drift_ppm"], 1000, delta=20)
        for line, cue in zip(lines, cues):
            self.assertAlmostEqual(line.start, cue.start + 2.5 + 0.001 * cue.start, delta=0.01)

    def test_lines_follow_a_jump(self):
        cues, words = film(offset=0.0, drift=0.0, jump_at=1200, jump=3.0)
        lines, _ = sync.align(cues, words)
        late = [(ln, c) for ln, c in zip(lines, cues) if c.start > 1300]
        self.assertTrue(all(abs(ln.start - (c.start + 3.0)) < 0.01 for ln, c in late))

    def test_ocr_slips_take_the_heard_word(self):
        self.assertEqual(
            sync.correct("The radio ts still dead.", ["the", "radio", "is", "still", "dead"]),
            "The radio is still dead.",
        )
        self.assertEqual(
            sync.correct("Imagine what we fold them.", ["imagine", "what", "we", "told", "them"]),
            "Imagine what we told them.",
        )

    def test_short_ocr_slips_become_i(self):
        self.assertEqual(
            sync.correct(
                "! enjoy working with people.", ["i", "enjoy", "working", "with", "people"]
            ),
            "I enjoy working with people.",
        )
        self.assertEqual(
            sync.tidy("...whether you might be having second thoughts..."),
            "Whether you might be having second thoughts.",
        )

    def test_lines_without_subtitles(self):
        words = [
            {"start": 0.0, "end": 0.3, "word": "Hello.", "prob": 1},
            {"start": 2.0, "end": 2.3, "word": "Dave", "prob": 1},
        ]
        self.assertEqual(len(sync.from_transcript(words)), 2)


class LabelTest(unittest.TestCase):
    def test_bracket_tags(self):
        self.assertTrue(subtitles.is_speaker_tag("HAL"))
        self.assertTrue(subtitles.is_speaker_tag("HAL SINGS"))
        self.assertTrue(subtitles.is_speaker_tag("MICHAELS"))
        for sound in (
            "GRUNTING",
            "ALARM BLARING",
            "APE MEN SCREAM",
            "INAUDIBLE DIALOGUE",
            "BEEPING NOISE",
        ):
            self.assertFalse(subtitles.is_speaker_tag(sound), sound)

    def test_bracket_label_alone_on_its_line(self):
        self.assertEqual(
            subtitles.split_speakers("[HALVORSEN]\nYou're welcome."),
            [("HALVORSEN", [], "You're welcome.")],
        )

    def test_italics_are_tagged(self):
        self.assertIn("italic", subtitles.split_speakers("<i>I'm sorry, Dave.</i>")[0][1])

    def test_labels_move_between_sources_over_the_same_words(self):
        words = [
            {"start": float(i), "end": i + 0.5, "word": w, "prob": 0.9}
            for i, w in enumerate("good evening dave how are you".split())
        ]
        plain = [
            subtitles.Cue(0.0, 2.6, "Good evening, Dave.", cue=0),
            subtitles.Cue(3.0, 5.6, "How are you?", cue=1),
        ]
        labelled = [
            subtitles.Cue(2.0, 4.6, "Good evening, Dave.", speaker="HAL", cue=0),
            subtitles.Cue(5.0, 7.6, "How are you?", speaker="BOWMAN", cue=1),
        ]
        a = ("plain", *sync.align(plain, words))
        b = ("labelled", *sync.align(labelled, words))
        name, lines, report, _ = sync.choose([a, b])
        self.assertEqual([ln.speaker for ln in lines], ["HAL", "BOWMAN"])
        self.assertEqual(report["labels_transferred"], 2)


class ScreenTest(unittest.TestCase):
    def test_long_lines_split_at_the_longest_pause(self):
        from voicebuild import quality

        words = [
            {"start": s, "end": s + 0.4, "word": "w", "prob": 0.9}
            for s in (
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                9.0,
                9.5,
                10.0,
                10.5,
                11.0,
                11.5,
                12.0,
                12.5,
            )
        ]
        line = sync.Line(0.0, 12.9, "x", "x", "HAL", [], 0, 1.0, words)
        parts = quality.split_long(line)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1].start, 9.0)

    def test_anchor_phrases(self):
        from voicebuild import speaker

        self.assertTrue(speaker._is_anchor("I'm sorry, Dave. I'm afraid I can't do that."))
        self.assertTrue(
            speaker._is_anchor("Well, this mission is too important for me to allow it.")
        )
        self.assertFalse(speaker._is_anchor("Open the pod bay doors, Hal."))


class UsableTest(unittest.TestCase):
    def test_muffled_and_noisy_lines_are_set_aside(self):
        import json
        import tempfile

        from voicebuild import build

        with tempfile.TemporaryDirectory() as d:
            film = Path(d, "film.mkv")
            b = build.Build(films=[film], subtitles=[], work=Path(d, "work"))
            hf = {"a": -18.0, "b": -20.0, "c": -21.0, "news": -42.0, "hiss": -19.0}
            kept = [{"id": k, "text": k} for k in hf]
            cleaned = {
                k: {"snr_db": 2.0 if k == "hiss" else 30.0, "hf_db": v, "path": "", "seconds": 2}
                for k, v in hf.items()
            }
            b.path(build.slug(film), "lines.json").write_text(json.dumps({"kept": kept}))
            b.path(build.slug(film), "clean", "clean.json").write_text(json.dumps(cleaned))
            ok, why = build.usable(b)
        self.assertEqual(sorted(ok), ["a", "b", "c"])
        self.assertTrue(why["news"].startswith("muffled"))
        self.assertTrue(why["hiss"].startswith("too noisy"))


class MonologueTest(unittest.TestCase):
    def line(self, start, end, text, speaker=None):
        words = [{"start": start, "end": end, "word": w, "prob": 0.9} for w in text.split()]
        return sync.Line(start, end, text, text, speaker, [], 0, 1.0, words)

    def test_pieces_of_a_monologue_join_with_their_pause(self):
        from voicebuild import quality

        lines = [
            self.line(0.0, 0.5, "Yes."),
            self.line(1.6, 4.0, "And it will stay that way."),
            self.line(4.5, 5.0, "Hmm.", "BOWMAN"),
            self.line(5.2, 7.0, "Of course."),
        ]
        verdicts = [
            {"hal": True, "sim": 0.8},
            {"hal": True, "sim": 0.7},
            {"hal": False, "sim": 0.3},
            {"hal": True, "sim": 0.8},
        ]
        runs = quality.join_runs(lines, verdicts)
        self.assertEqual(
            [r[0].text for r in runs], ["Yes. And it will stay that way.", "Of course."]
        )
        self.assertEqual((runs[0][0].start, runs[0][0].end), (0.0, 4.0))

    def test_clip_names_come_from_the_words(self):
        from voicebuild import build

        taken = set()
        self.assertEqual(
            build.clip_name("I'm sorry, Dave. I'm afraid I can't do that.", taken),
            "im-sorry-dave-im-afraid-i-cant-do-that",
        )
        self.assertEqual(build.clip_name("None whatsoever, Frank.", taken), "none-whatsoever-frank")
        self.assertEqual(
            build.clip_name("None whatsoever, Frank!", taken), "none-whatsoever-frank-2"
        )


class SignatureLineTest(unittest.TestCase):
    def build(self, d, hold_out):
        import json

        from voicebuild import build

        film = Path(d, "film.mkv")
        b = build.Build(films=[film], subtitles=[], work=Path(d, "work"), hold_out=hold_out)
        texts = {
            "sorry": "I'm sorry, Dave. I'm afraid I can't do that.",
            "other": "Good evening, Dave.",
        }
        cleaned = {}
        for k in texts:
            wav = Path(d, f"{k}.wav")
            wav.write_bytes(b"")
            cleaned[k] = {"snr_db": 30.0, "hf_db": -20.0, "path": str(wav), "seconds": 2}
        kept = [{"id": k, "text": v} for k, v in texts.items()]
        b.path(build.slug(film), "lines.json").write_text(json.dumps({"kept": kept}))
        b.path(build.slug(film), "clean", "clean.json").write_text(json.dumps(cleaned))
        build.dataset(b)
        return b.path("dataset", "transcripts.csv").read_text().splitlines()

    def test_signature_line_is_trained_on_and_weighted(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            rows = self.build(d, hold_out=False)
        self.assertIn("sorry,\"I'm sorry, Dave. I'm afraid I can't do that.\",3", rows)

    def test_hold_out_leaves_it_out(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            rows = self.build(d, hold_out=True)
        self.assertEqual(rows, ['other,"Good evening, Dave.",1'])


class InputTest(unittest.TestCase):
    def test_films_and_subtitles_found_in_a_folder(self):
        import tempfile
        from voicebuild.__main__ import films_in

        with tempfile.TemporaryDirectory() as d:
            for name in ("film.mkv", "film.en.srt", "notes.txt.bak", "cover.jpg"):
                Path(d, name).write_text("")
            films, subs = films_in([Path(d)])
        self.assertEqual([f.name for f in films], ["film.mkv"])
        self.assertEqual([s.name for s in subs], ["film.en.srt"])


if __name__ == "__main__":
    unittest.main()
