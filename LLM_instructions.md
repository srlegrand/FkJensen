# Instructions for AI assistants

The one instruction file for any AI assistant working on this repository: what to check before starting, and how changes are delivered. Read this file and `DESIGN_PHILOSOPHY.md` before working here, and follow both. With every change, update `README.md` and the pages in `docs/` that describe what changed.

## Checks to run first

At the start of a session, run these and report each to the person as passed or failed, with the fix when it failed.

1. **Design philosophy.** Read `DESIGN_PHILOSOPHY.md` and tell the person it exists. When a request goes against it, say which point and ask whether to follow the file or update it.
2. **A workspace that runs commands** and keeps files between turns. Without it you can only advise.
3. **The program builds and its tests pass.** Clone the repository, `cargo test --release`, and `cargo test --release` in `tools/fkj-import`. No system libraries are needed. Report how many tests passed. The built-in set is the ground and mountains of `src/world.rs` with the T-rex and the car from `assets/models/`, stored with Git LFS (`git lfs install` before cloning, or `git lfs pull` after); the tests do not need them.
4. **GitHub access.** Check that you can read `srlegrand/FkJensen` (private: the Claude GitHub App must be installed on Simon's account with access to it). If it cannot be read, ask the person to attach the repository or to send the current `main` as a tarball. Do not push: the person pushes.
5. **Report** one list: what passed, what failed, what the person has to do.

## The loop

1. The person asks for a change in plain words.
2. The assistant works in its own clone. Before every change it fetches `main` and starts from it. It builds, runs `cargo test --release`, renders the built-in scene headless (`--png`) and looks at the image, and updates `README.md` and `docs/`. It fetches `main` again just before making the bundle and rebases if anything came in.
3. The assistant makes one git bundle of its commits since `origin/main`, authored and committed by Simon Legrand, on a branch named `fkjensen-update`: `git bundle create fkjensen-<what>-<DDMMYYYY-HHMM>.bundle origin/main..fkjensen-update`. The suffix is new every time, so the chat shows a new download. A bundle carries whole commits (renames, deletions, binary files, Git LFS pointers) and holds everything not yet on GitHub, so any machine at `origin/main` takes it whole, whatever was sent before. The assistant checks it with `git bundle verify` and by fast-forwarding a clean copy of `origin/main` to it.
4. The assistant sends the bundle and any screenshot worth looking at, with the two blocks of commands below filled in, and says what changed, what was tested and what was not. One bundle and three blocks per response, every time: one for the local machine, one for the Raspberry Pi, one for the Windows machine.
5. The person clicks "Download all" and pastes the local block on whichever machine they are at, the desktop or the laptop:

       cd ~/Software/FkJensen && \
       git fetch origin && \
       git switch main && \
       git branch -f before-update && \
       git reset --keep origin/main && \
       git pull --ff-only ~/Downloads/fkjensen-<what>-<suffix>.bundle fkjensen-update && \
       git lfs pull && \
       cargo build --release && cargo test --release && \
       git push origin main && \
       cargo run --release

   The commands are joined with `&&`: a bundle that does not fast-forward is never pushed. `before-update` keeps whatever the machine had committed and not pushed; `reset --keep` stops rather than lose changes not committed. The block holds no other commands; deletions and renames travel in the bundle. Once pushed, every other machine takes it with `git pull`. The exceptions: when the assistant sends `.fja` models too large for the chat's limit, the block moves them from `~/Downloads` into `assets/models/` and commits them; when a change to `tools/fkj-import` or the format means the models must be imported again, the block builds the importer and runs it on the source files (commands in `README.md`) before the commit. `cargo run` comes last so the program can stay open.

   Then the Pi block, also on the desktop, after the local block has pushed. It cross-compiles for the Pi 1 (ARMv6, 32-bit), copies the binary over and runs it there headless, so the output comes back to the desktop terminal:

       cd ~/Software/FkJensen && \
       cross build --release --target arm-unknown-linux-gnueabihf && \
       scp target/arm-unknown-linux-gnueabihf/release/fkjensen simon@RPi-1:~/ && \
       ssh simon@RPi-1 'mkdir -p ~/assets/models' && \
       scp assets/models/trex.fja assets/models/car.fja simon@RPi-1:~/assets/models/ && \
       ssh simon@RPi-1 './fkjensen --hardware && ./fkjensen --png pi.png' && \
       scp simon@RPi-1:~/pi.png ~/Downloads/fkjensen-pi.png

   The low-resolution models go to `~/assets/models/` on the Pi, beside the binary, where the program looks for them (the high-resolution T-rex never fits a Pi); the two lines can be left out when they have not changed. The `ssh` line holds whatever the change needs tested on the Pi; it always starts with `--hardware`. The rendered image comes back to `~/Downloads` to be looked at or attached.
   Then the Windows block, in PowerShell on the Windows machine (Git, Git LFS, the MSVC build tools and Rust installed), after the local block has pushed. It builds natively from GitHub's `main`:

       cd $HOME\Software
       if (Test-Path FkJensen) { git -C FkJensen checkout main; git -C FkJensen pull --rebase origin main; git -C FkJensen lfs pull } else { git lfs install; git clone -b main https://github.com/srlegrand/FkJensen.git }
       cd FkJensen
       cargo test --release
       cargo run --release -- --hardware
       cargo run --release -- --png $HOME\Downloads\fkjensen-win.png --debug-view all
       cargo run --release -- --visual-debug

6. When the pull from the bundle fails, the person pastes the terminal output; the assistant fetches `main`, rebases, tests again and sends a new bundle.
7. Until the person says a bundle is pushed, the assistant keeps its work committed locally and makes the next bundle on top of it, holding everything since `origin/main`; it fetches `origin/main` first and rebases onto it when it has moved.

## Conventions

- Tests with every feature; `cargo test --release` passes, in the repository and in `tools/fkj-import`, before a change is handed over.
- Published models live in `assets/models/`, credited in `assets/README.md`. Models that must not be published go in `assets/local/`, which git ignores.
- Documentation with every change: `README.md`, and `docs/` pages for anything with a page. Every page has a "Not done yet" list.
- A screenshot in `docs/` when something visible changed, taken from `--png` or from the running program.
- To explain or check a stage, render its visual debug view (`--png out.png --debug-view NAME`, or `all`) and look at it. Those views are 64-bit only and are never a reason to hold back the Raspberry Pi build; a new stage gets a view.
- Say what was tested and what was not. Numbers in the docs are measured and say on what hardware.
- No GPU API, ever: no wgpu, Vulkan, OpenGL, Metal, DirectX, no shader files. The frame reaches the screen through softbuffer. A dependency that pulls a GPU API in is a bug.
- Dates DD/MM/YYYY. No em dashes.
- Files under 100 MB in the repository, bundles under 30 MB in the chat.
- Credit: FkJensen was created by Simon Legrand. No attribution to AI tools anywhere in the repository, commits or pull requests.
