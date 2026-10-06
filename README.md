# A-SSTV-Encoder-and-Decoder-by-dsh-dsh-SSTV-
**SSTV Studio 1.2.0** (released 2026-02-14) · `python sstv_app.py --version` for
the full history

A Windows program for sending and receiving **SSTV** (slow-scan television) —
the mode radio amateurs use to send still pictures over a voice channel.

It does both directions, one at a time (SSTV is a simplex mode):

| | Input | Output |
|---|---|---|
| **Transmit** | an image file, the built-in test card, or whatever you last received | the speakers, or a WAV file |
| **Receive** | the microphone, the **system sound**, or a WAV file | a picture on screen, drawn as it arrives, saved as PNG or JPEG |

---

## Running it

Double-click **`run.bat`**, or from a command prompt:

```
python sstv_app.py
```

To check the installation without opening a window — this encodes and decodes
every supported mode and reports the result:

```
run.bat --check
```

There are also two command-line modes, useful for scripting or for decoding a
recording without the interface:

```
run.bat --encode picture.png --mode "Robot 36" --out sent.wav
run.bat --decode recording.wav          # writes recording-decoded.png
```

---

## Supported modes

All fourteen of the mainstream modes. The transmission time includes the VIS
header.

| Mode | Picture | Line time | Takes |
|---|---|---|---|
| Robot 36 | 320 × 240 | 150.000 ms | 36.9 s |
| Robot 72 | 320 × 240 | 300.000 ms | 72.9 s |
| Martin M1 | 320 × 256 | 446.446 ms | 1 m 55 s |
| Martin M2 | 320 × 256 | 226.798 ms | 59.0 s |
| Scottie S1 | 320 × 256 | 428.220 ms | 1 m 50 s |
| Scottie S2 | 320 × 256 | 277.692 ms | 1 m 12 s |
| Scottie DX | 320 × 256 | 1050.300 ms | 4 m 30 s |
| PD 50 | 320 × 256 | 388.160 ms | 50.6 s |
| PD 90 | 320 × 256 | 703.040 ms | 1 m 31 s |
| PD 120 | 640 × 496 | 508.480 ms | 2 m 07 s |
| PD 160 | 512 × 400 | 804.416 ms | 2 m 42 s |
| PD 180 | 640 × 496 | 754.240 ms | 3 m 08 s |
| PD 240 | 640 × 496 | 1000.000 ms | 4 m 09 s |
| PD 290 | 800 × 616 | 937.280 ms | 4 m 50 s |

If you are not sure which to use: **Robot 36** is the fastest and is the usual
choice for a quick contact; **PD 120** is the usual choice for a good picture in
about two minutes. The receiver reads the mode from the VIS header, so it does
not need to be told in advance.

---

## Using it

### Sending a picture

1. Open the **Transmit** tab.
2. **Open image...** — any PNG, JPEG, BMP, GIF or TIFF. It is resized to the
   mode's exact picture size for you.
3. Pick a **Mode**.
4. Check the **Send via** device, then press **Transmit**.

The audio is prepared as soon as you load an image, so pressing Transmit starts
straight away. The progress bar tracks the transmission and **Stop** aborts it
mid-picture.

**Save audio file...** writes the transmission to a WAV file instead of playing
it — useful for sending it over a radio yourself, for a digital-mode repeater, or
for keeping a copy.

Keep the playback volume at a level that does not distort: SSTV's own levels sit
at about 80% of full scale, so a sound card that is set too loud will clip and the
far end will see a torn picture. The **Receive** tab's level meter shows you what
a signal looks like when it arrives cleanly.

### Receiving a picture

1. Open the **Receive** tab.
2. Choose where the signal comes from:
   * **Microphone or line input** — a microphone, or a receiver wired into the
     sound card's input.
   * **System sound (what is playing)** — whatever this computer is sending to its
     speakers. Use this when the receiver is feeding the speakers, or when the
     signal is playing in another program such as a WebSDR in a browser. Nothing
     needs to be plugged in or looped back with a cable.
3. Press the record button and press it again when the transmission ends, or press
   **Open audio file...** to decode a recording.
4. The picture appears as it is decoded, line by line, and the **waterfall**
   underneath scrolls the incoming signal in real time.

System-sound listening uses WASAPI loopback, so it works on any machine with
working audio. It does not depend on the *Stereo Mix* input, which is hidden and
disabled by default on most modern sound drivers.

Leave **Detect mode from VIS header** ticked and you do not need to know which
mode is being sent. The panel under the picture reports what was received:

* the mode and its VIS code
* how many lines were decoded, and how confident the header read was
* whether the decoder had to correct a tuning offset, and by how much
* the accuracy of the synchronising pulses it found

### When the header was missed

If you start listening after a transmission has already begun, its VIS header is
gone — it is sent once, at the very start. Rather than giving up, the program works
the mode out from the signal itself: every SSTV mode repeats a synchronising pulse
once per line, and the line period is what tells the modes apart, so the rhythm of
those pulses identifies the mode. All 14 modes are recognised this way.

The panel then says the mode was **inferred**, and how confident the inference is,
because a guess is not the same as a reading. Two things are worth knowing:

* **Framing.** SSTV has no vertical sync, so which line ends up at the top is
  arbitrary to begin with. An inferred decode can also sit a little way to one side,
  because the line grid is measured from the signal rather than taken from the
  header's known timing. The picture is readable; it may not be pixel-perfect.
* **Quality.** With a header the decode is exact. Inferred decoding is close in the
  PD, Martin and Robot families and looser in the Scottie family, whose sync pulse
  is the shortest.

If you know the mode, choosing it in the **Mode** list is better than letting it be
inferred: the decoder then uses the exact timing and no guessing is involved.

**Save image...** writes the result as PNG or JPEG.

### Reading the waterfall

The waterfall plots frequency across and time downwards, with brightness for
strength — so you can see what is arriving before a single line of picture has
been decoded. SSTV is easy to recognise by eye once you know what to look for:

| What you see | What it means |
|---|---|
| A bright band covering about a third of the width, crossed by a **regular comb of bright vertical lines** | A healthy SSTV signal. The comb is the line sync pulses, and even spacing means it will decode. |
| A short, busy pattern at the very start | The VIS header — the tones that name the mode. |
| A steady thin line that never moves | A carrier or a whistle, not a picture. |
| Brightness spread across the **whole width** | Too loud: the audio is clipping and the picture will be torn. Turn the volume down. |
| A faint haze with no structure | Noise only. There may be a signal in there, but not enough of one. |
| The bright band well to one side of the `sync` marker | The receiver is off frequency. The decoder corrects a modest offset and reports the amount. |

The scale along the top labels the four tones that matter — `sync` 1200 Hz,
`black` 1500 Hz, `mid` 1900 Hz, `white` 2300 Hz — so a tuning error shows up as
the band shifting relative to those marks. **Frequency span** zooms in on the
band, **History** sets how many seconds are shown, and **Suggested history for
mode** picks a sensible length for whatever mode you are expecting. Untick
**Show frequency scale** for a plain picture.

The waterfall is drawn while recording, and is also filled in when you open an
audio file, so a recording can be inspected after the fact.

The picture and the waterfall share the column through a draggable divider — drag
it to give either one more room. The window sizes itself to the monitor it opens
on, so it will not spread across two screens, and if it is shorter than the
controls need, the control column scrolls rather than hiding the buttons at the
bottom.

---

## If something goes wrong

| Symptom | What it means |
|---|---|
| "No SSTV signal found" | The audio has no VIS header — it may be speech, noise, or the middle of a transmission rather than its start. |
| The picture is torn or slanted | The received signal was weak or the audio clipped. Try a lower playback volume. |
| Colours are wrong but the shape is right | Usually a tuning offset; the decoder corrects modest ones automatically and reports what it corrected. |
| "Partly decoded ..." | The recording ended before the transmission did. |
| Nothing appears in the device lists | Windows has no audio device enabled that the program can see. Check the sound settings. |
| The waterfall shows no bright band at all | No signal is reaching the input. Check the radio's volume, the cable, and that the right input device is selected. |

---

## Notes on the implementation

* **No installation step.** The program depends on NumPy and Pillow, which are
  already present in the `vendor/` folder alongside it; `sstv_bootstrap.py` puts
  that folder on the import path. Only Python itself and its standard library
  need to be present, and `tkinter` (bundled with the python.org installer).
* **Audio goes straight to Windows.** Playback and recording use the Win32
  multimedia API through `ctypes`, so there is no PortAudio or PyAudio
  dependency to install. WAV files are read and written with the standard
  library.
* **Verification.** `run.bat --check` encodes and decodes every mode and prints
  the peak signal-to-noise ratio of each round trip, and also checks the
  waterfall. Decoding is additionally verified against added noise, against a
  signal attenuated to 8% of full scale, and against a deliberate tuning offset.

### Sound rates

Everything works at 48 000 Hz, which is what QSSTV and PySSTV use and what
sound cards handle natively. Recording from a device that only offers 44 100 Hz
is resampled automatically.

---

## Version history

`python sstv_app.py --version` prints this from the program itself.

### 1.1.0 — 2026-02-13

* **Receive from the system sound.** A receiver feeding the speakers, or a signal
  playing in another program such as a WebSDR, can now be decoded with no cable
  between the two. This uses WASAPI loopback, so it does not depend on the *Stereo
  Mix* input that most modern sound drivers hide and disable.
* **The picture is drawn as it arrives.** Instead of waiting until recording stops
  before decoding anything, lines appear from about a second in and the picture
  fills down the screen. Stopping no longer begins a long decode.
* **The waterfall is no longer cleared by a decode.** Its history is kept, so the
  signal you were watching stays on screen.
* **Window sizing follows the monitor it is on**, so the window no longer spreads
  across two screens, and the control column scrolls when the window is short so
  the buttons at the bottom remain reachable.

### 1.0.0 — 2026-02-12

* Encode and decode all 14 mainstream SSTV modes.
* Simplex operation: transmitting and receiving are separate actions.
* Receive from a microphone or an audio file; transmit to the speakers or an audio
  file.
* Real-time waterfall while recording, with a frequency scale and a history length
  suited to each mode.
* Decode without being told the mode, by reading the VIS header.

### Versions

The version lives in exactly one place — `sstv/version.py` — and everything else
reads it: the window title, `--version`, the self check, both READMEs and the
package folder name. So a release cannot end up half-labelled.

To move to the next version:

```
python tools\bump_version.py patch     # 1.1.0 -> 1.1.1   (a fix)
python tools\bump_version.py minor     # 1.1.0 -> 1.2.0   (a new capability)
python tools\bump_version.py major     # 1.1.0 -> 2.0.0   (a breaking change)
python tools\bump_version.py 1.4.2     # or set it outright
```

It updates the number and the release date everywhere they appear, prints the
changelog entry to paste into `sstv/version.py`, and refuses to leave a
disagreement behind — `python tools\bump_version.py --check` reports any file
whose version or date is stale.

---

## Layout

```
run.bat                 launcher
sstv_app.py             entry point: window, --check, --encode, --decode
sstv_bootstrap.py       puts vendor/ on the import path
sstv/
    modes.py            the scan plan of every mode: timing, tones, VIS codes
    encoder.py          image  -> audio
    decoder.py          audio  -> image
    dsp.py              FM modulation, demodulation, filtering, spectrum analysis
    waterfall.py        the scrolling spectrum display
    colorspace.py       RGB <-> YCrCb conversion
    vis.py              reading the mode identifier off the air
    audio.py            WAV files, speaker playback, microphone capture
    wasapi.py           capturing the system sound (WASAPI loopback)
    streaming.py        decoding as the transmission arrives
    version.py          the version number and its history
    gui.py              the window
tests/                  the verification scripts described above
```
