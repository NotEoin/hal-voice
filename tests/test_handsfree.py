"""Hands-free listening: who is spoken to, where utterances end, and the voice test's scoring."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class AddressedTest(unittest.TestCase):
    def check(self, text, named, request):
        from halvoice.handsfree import addressed

        self.assertEqual(addressed(text), (named, request), text)

    def test_name_first_and_last(self):
        self.check("Hal, what time is it?", True, "what time is it")
        self.check("What time is it, Hal?", True, "What time is it")
        self.check("Hall, open Firefox.", True, "open Firefox")

    def test_greeting_before_the_name_is_kept_for_a_reply(self):
        self.check("Good morning Hal, what's the weather today?", True,
                   "Good morning. What's the weather today")
        self.check("Evening Hal, give me the evening briefing.", True,
                   "Evening. Give me the evening briefing")
        self.check("Good morning, Hal.", True, "Good morning")
        self.check("Thanks Hal.", True, "Thanks")

    def test_attention_word_alone_waits_for_the_request(self):
        self.check("Hey Hal.", True, "")
        self.check("Hey Hal, set a timer.", True, "set a timer")
        self.check("Hal.", True, "")

    def test_how_is_the_name_only_where_it_cannot_start_a_sentence(self):
        self.check("How, what time is it?", True, "what time is it")
        self.check("How? Stop.", True, "Stop")
        self.check("How what's on my calendar?", True, "what's on my calendar")
        self.check("How hot is my graphics card running?", False,
                   "How hot is my graphics card running")
        self.check("How?", False, "How")

    def test_talk_about_hal_is_not_to_him(self):
        self.check("Hal said he'd ring me back tomorrow.", False,
                   "Hal said he'd ring me back tomorrow")
        self.check("I'll ask Hal about it later.", False, "I'll ask Hal about it later")
        self.check("Good morning, Sam.", False, "Good morning, Sam")


class SegmenterTest(unittest.TestCase):
    def frames(self, seconds, rng):
        """Fan-like noise: a strong hum and rumble with broadband hiss, as int16 frames."""
        sr, n = 16000, int(16000 * seconds)
        t = np.arange(n) / sr
        y = 0.25 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 120 * t)
        y += 0.03 * rng.standard_normal(n)
        pcm = np.clip(y * 32768, -32768, 32767).astype(np.int16)
        return [pcm[i: i + 480] for i in range(0, n - 480, 480)]

    def test_steady_fan_noise_is_not_an_endless_utterance(self):
        from halvoice.handsfree import Segmenter

        rng = np.random.default_rng(1)
        seg = Segmenter(16000, 2, 600, 2.0, highpass_hz=300, margin_db=8)
        out = [seg.feed(f) for f in self.frames(6, rng)]
        self.assertEqual([o for o in out if o is not None], [])
        self.assertEqual(seg.voiced, [])


class OverlapTest(unittest.TestCase):
    def ears(self, quiet_since, now):
        from halvoice import handsfree

        e = handsfree.Ears(lambda a: "")
        e.clock = lambda: now
        e._quiet_since = quiet_since
        e.end_ms = 600
        return e

    def test_echo_of_hal_is_overlapped(self):
        e = self.ears(100.0, 101.0)
        audio = np.zeros(int(0.8 * 16000), dtype=np.float32)  # 100.2 to 101.0
        self.assertTrue(e.overlapped(audio, 0.0))

    def test_reply_that_starts_in_the_echo_tail_is_not(self):
        # Breath at 100.4, then a reply running to 102.4: the user, not HAL's echo.
        e = self.ears(100.0, 103.0)
        audio = np.zeros(int(2.6 * 16000), dtype=np.float32)
        self.assertFalse(e.overlapped(audio, 0.0))


class ScriptTest(unittest.TestCase):
    def test_lines_and_what_they_should_do(self):
        from voicetest import script

        lines = {ln.id: ln for ln in script.load()}
        self.assertEqual(sum(ln.kind == "test" for ln in lines.values()), 165)
        self.assertTrue(lines["2"].then_request)
        self.assertEqual(lines["15"].request, "Good morning. What's the weather today")
        self.assertFalse(lines["9"].turn)
        self.assertTrue(lines["12"].turn)
        self.assertFalse(lines["33"].wake)
        self.assertTrue(lines["40"].followup)

    def test_numbers_and_spellings_compare_equal(self):
        from voicetest import score

        self.assertEqual(score.wer("Hal, add a dentist appointment at 10am",
                                   "Hal, add a dentist appointment at ten a.m."), 0.0)
        self.assertEqual(
            score.wer("convert \u00a350 to euros", "convert fifty pounds to euros"), 0.0)
        self.assertEqual(score.cut("take a note", "take a note blog post idea"), "end")


if __name__ == "__main__":
    unittest.main()
