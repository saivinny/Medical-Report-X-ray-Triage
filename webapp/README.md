# Web demo

`triage-assistant.html` is a standalone, browser-only interactive mock of the TriageNet
pipeline — no install, no server, just open it in a browser. It's meant for the live Week 11/12
pitch/demo and for anyone reviewing the project who doesn't want to set up the Python
environment.

It uses simple JavaScript heuristics (pixel-contrast for the image stream, keyword-weighting
for the text stream) to *simulate* what the real trained models in `../src/` would output. The
real, trainable models are the ones in the main repository — this file is presentation only.

Open it by double-clicking, or:

```bash
python -m http.server 8000   # from this folder
# then visit http://localhost:8000/triage-assistant.html
```
