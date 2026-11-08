# Local TTS visualiser (web UI)

`webui/` is a local web page for live text-to-speech with trained checkpoints. It needs only the notebook environment (PyTorch, TorchAudio, NumPy) and Python's standard-library HTTP server. With the project environment active, run from the repository root:

```sh
python webui/server.py
```

The server loads the default checkpoint, serves `http://127.0.0.1:8000/` and opens it in the browser. The page has two tabs.

**Agent mode** demonstrates TTS for a voice agent ([docs/VOICE_AGENT_TTS.md](VOICE_AGENT_TTS.md)). A simulated language model streams a reply word by word, after a set time to first token and at a set speed; no real LLM runs, so a run is offline and repeatable. Up to three strategies speak the same reply at the same time with real TTS calls: the whole sentence, independent N-word chunks, or context-aware N-word chunks with lookahead K (the conditions of the chunk study; `/api/synthesize_chunk` uses the same synthesis as `tools/chunk_study.py`). Chunks are queued back to back with Web Audio; one strategy is audible during the run, and afterwards each can be replayed, alone or as an A/B pair. The timeline shows when words arrive, each TTS call and the playback, with the time to first audio and stalls (the audio ran out before the next chunk was ready). The result cards show those live measurements next to each strategy's Whisper WER from the chunk study (`docs/records/chunk-study/R0_40k/analysis.json`), which is never computed live; two charts show the study's WER against waiting and how the waiting saved depends on the LLM speed and the sentence length. The three strategies share one GPU, so their calls queue behind each other.

**Reader** reads any English text: the text is split into sentences, each sentence is synthesised and played, and the current sentence and word are highlighted. A second checkpoint can be selected for A/B comparison. For the current sentence the page shows the mel spectrogram, the text-to-frame alignment returned by `model.infer(...)["attention"]` and the inference time; it also shows each checkpoint's measured WER and downloads the read text as one WAV file.

The model list is `webui/config.example.json`. Paths are relative to the repository root and may be glob patterns (the newest match is used); a checkpoint that is not found is listed as unavailable. To change the list, copy the file to `webui/config.json` (git-ignored) and edit the labels, paths, descriptions and WER values. A new checkpoint, such as a chunk-aware fine-tune, is one more entry in `models`; its chunk-study results appear in Agent mode once an entry in `chunk_studies` points to its `analysis.json` (entries whose file does not exist are skipped). `agent` sets the preset replies, the default strategies and the default stream settings. Each checkpoint is rebuilt from the switches and speaker configuration stored in it, so a speaker-conditioned checkpoint gets a speaker selector; it must use the notebook's symbol table. Text preparation (`webui/text_en.py`) writes out numbers, money, times, percentages and common abbreviations in the LJSpeech style; it does not handle acronyms, roman numerals, units, dates with slashes or URLs.

Options: `--port`, `--device cpu`, `--config PATH`, `--no-browser`, and `--host` (default `127.0.0.1`; the API has no authentication).
