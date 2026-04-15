# Step1 Setup

This document only covers the public `step1` download path:

- `yt-dlp`
- `bgutil-ytdlp-pot-provider`
- optional cookies
- `python -m pipeline.step1_download`

It does not cover `step2-step4`.

## Prerequisites

- Downloading videos from overseas platforms requires working network access.
- Python 3.11+ recommended
- `ffmpeg` and `ffprobe` on `PATH`
- Node.js 18+ on `PATH`
- `git`

## 1. Install Python dependencies

Create a virtual environment and install the repo requirements:

```bash
cd /path/to/Translate_Open
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

If you only want the step1 download chain, you can install the smaller dependency set instead:

```bash
python -m pip install -r requirements_download.txt
```

## 2. Install or update yt-dlp

The bgutil provider requires a recent `yt-dlp`. The upstream provider README says `yt-dlp 2025.05.22 or above` is required.

If you use pip:

```bash
python -m pip install -U yt-dlp
```

If you prefer the standalone binary, follow the official `yt-dlp` installation guide instead.

## 3. Install the bgutil yt-dlp plugin

If `yt-dlp` is installed by `pip` or `pipx`, the simplest path is:

```bash
python -m pip install -U bgutil-ytdlp-pot-provider
```

Verification:

```bash
yt-dlp -v "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Look for verbose output that mentions the external bgutil providers.

## 4. Install the bgutil provider service

This repository assumes the HTTP provider mode, because it is simpler for public users and avoids per-download script startup overhead.

Clone the provider source and build it:

```bash
cd ~
git clone https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git
cd bgutil-ytdlp-pot-provider/server
npm install
npx tsc
```

This should produce:

```text
/path/to/bgutil-ytdlp-pot-provider/server/build/main.js
```

You do not need to start it manually if `video.pot_provider=bgutil_http`, because `pipeline.step1_download` will auto-start it when needed.

## 5. Configure `config.yaml`

Start from the example:

```bash
cd /path/to/Translate_Open
cp config.example.yaml config.yaml
```

For bgutil HTTP mode, set:

```yaml
video:
  url: https://www.youtube.com/watch?v=YOUR_VIDEO_ID
  auth_method: none
  pot_provider: bgutil_http
  pot_base_url: http://127.0.0.1:4416
  bgutil_provider_root: /path/to/bgutil-ytdlp-pot-provider
```

If the provider is installed somewhere else, change `bgutil_provider_root`.

## 6. Optional cookies setup

Cookies are only needed when anonymous access is unstable or restricted for the target video or playlist.

Two supported modes are exposed in this repository:

- `auth_method: browser`
- `auth_method: cookies_file`

### Browser cookies

Set:

```yaml
video:
  auth_method: browser
  cookies_from_browser: chrome
```

Use this only when the local machine already has a logged-in browser profile that `yt-dlp` can read.

### Cookies file

Export a Netscape-format cookies file from your browser and point config at it:

```yaml
video:
  auth_method: cookies_file
  cookies_file: ./youtube_cookies.txt
```

Recommended handling:

- keep the cookies file outside the repository when possible
- add it to `.gitignore`
- rotate it when downloads start failing with login or bot-check errors

This open-source repository already ignores common cookie filenames.

## 7. Run step1

Single video:

```bash
cd /path/to/Translate_Open
source .venv/bin/activate
python -m pipeline.step1_download --workdir ./workdir --source "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

Playlist:

```bash
cd /path/to/Translate_Open
source .venv/bin/activate
python -m pipeline.step1_download --workdir ./workdir --source "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
```

## 8. What step1 writes

Under the target `--workdir`, step1 writes:

- `*.mp4`
- `*.info.json`
- thumbnails such as `*.webp`
- `downloaded_ids.json`

It does not require tracker databases, cron, or upload services.

## Troubleshooting

- `bgutil provider not found`
  Build the provider under `bgutil_provider_root/server` and make sure `build/main.js` exists.
- `node is required`
  Install Node.js 18+ and make sure `node` is on `PATH`.
- `Sign in to confirm you're not a bot`
  Confirm the plugin is installed, `pot_provider` is set to `bgutil_http`, and try cookies if anonymous mode is still blocked.
- `yt-dlp` does not show bgutil providers in verbose mode
  Reinstall `bgutil-ytdlp-pot-provider` into the same Python environment or `pipx` environment used by `yt-dlp`.
