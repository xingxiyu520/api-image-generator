---
name: api-image-generator
description: Generate, edit, or batch-process images through a configured OpenAI-compatible or Replicate image API, including local gateways. Use for 文生图, 图生图, reference-image edits, and batch image generation when outputs should be saved locally.
---

# API Image Generator

Use `scripts/generate_image.py` from this skill directory. Pass the requested output directory, or use `outputs/` in the workspace. Prefer an existing API configuration when the user has already chosen a provider. Read secrets from environment variables or a local `.env` file. Keep keys out of chat, arguments, logs, and batch files.

## Generate, edit, and batch

```powershell
python scripts/generate_image.py --provider openai-compatible --prompt "A warm cabin illustration" --output-dir outputs
python scripts/generate_image.py --provider openai-compatible --prompt "Change the background to blue" --input-image reference.png --output-dir outputs
python scripts/generate_image.py --provider openai-compatible --prompt "A warm cabin illustration" --count 4 --output-dir outputs
python scripts/generate_image.py --provider openai-compatible --prompt "Replace the background with white" --input-dir references --output-dir outputs
python scripts/generate_image.py --provider openai-compatible --batch-file jobs.json --output-dir outputs
```

- `--input-image` takes an existing PNG, JPEG, or WebP. Repeat it for multiple references on a compatible OpenAI-style endpoint. With `--input-dir`, each supported file in the top-level folder gets its own request. Use a separate output directory to avoid processing generated files on later runs.
- `--count` sends individual requests sequentially. A manifest can mix generation and edits, with separate prompts and options per job. The job count determines how many API calls will be made.
- A single job returns `saved_path`. Batches return `succeeded`, `failed`, and `results`, with one record per attempted image. A failed request is recorded while the remaining jobs run. The CLI exits with status 1 if any batch item failed. Report successful paths and identify failed items before rerunning.

A UTF-8 `jobs.json` is a nonempty JSON array. Relative reference paths are resolved from the manifest's folder. Each job requires `prompt`. Optional fields are `input_images`, `count`, `model`, `size`, `negative_prompt`, and `seed`. Entries inherit command-line defaults for omitted options. Provider-specific parameters depend on the model.

```json
[
  {"prompt": "A warm cabin illustration", "count": 3},
  {"prompt": "Change the background to blue", "input_images": ["reference.png"]}
]
```

## API configuration

- **OpenAI-compatible:** Set `IMAGE_API_BASE_URL` to the provider's API root, for example `http://127.0.0.1:8000/v1`, and set `IMAGE_API_MODEL` or pass `--model`. Set `IMAGE_API_KEY` for authenticated endpoints. A loopback gateway can use a blank key if it allows unauthenticated requests. With no reference, the script uses `/images/generations`. With references, it uses `/images/edits` and uploads multipart files as `image` for one file or `image[]` for multiple files. Set the full `IMAGE_API_EDIT_URL` if the gateway uses a different compatible edit path. The configured gateway and model must accept the operation and fields used.
- **Replicate:** Set `REPLICATE_API_TOKEN` and `REPLICATE_MODEL` (`owner/model` or version hash). An editing job supports one reference and sends it as a data URL in the model's image input. Set `REPLICATE_IMAGE_FIELD` when that input is named something other than `image`. Use a model that accepts a single file input.

Use [.env.example](.env.example) for the variable names. The script loads an explicit `--env-file`, otherwise the workspace `.env` if present, otherwise this skill's `.env`. Exported process variables take precedence, and `--model` overrides the configured default for jobs without a per-job model. When switching gateways, update the address, model, and matching credential together. For offline validation, `--mock-response` accepts a local JSON fixture containing base64 image data.
