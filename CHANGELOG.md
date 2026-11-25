# Changes in this fork

This branch (`saros-port`) starts from the `saros` branch of
[CarRamRod-963/KytyPS5](https://github.com/CarRamRod-963/KytyPS5) (its `saros-v0.1.7` since
2026-11-17, `saros-v0.1.5` before) and adds the work below, most of it for Saros (PPSA07631) on
Windows. Several pieces are ported from, or follow ideas of, the
Demon's Souls fork [chenxiao07/KytyPS5](https://github.com/chenxiao07/KytyPS5); those are marked
"from the DS fork".

Numbers were measured on Saros with an RTX 4090 and an i9-13900K, each as an off/on comparison
inside one running session unless it says otherwise. They belong to the spot and settings named
with them; other places and machines will differ.

## For players

### A launcher for Windows

- `launcher.cmd` opens **KytyLauncher**, a settings window with pages for play, display, graphics,
  game, shaders and advanced options (from the DS fork). It needs no install; build it with
  `launcher-windows\build.cmd`. Without the app, `launcher.cmd` falls back to the script
  `launcher.ps1`, which offers the same settings.
- `run-windows.ps1` starts the emulator from a settings file or from options, and
  `build-windows.cmd` builds it. Both are what the launcher runs underneath.
- Settings are kept in `launcher-settings.json` next to the launcher (not tracked by git).

### Frame generation (DLSS Frame Generation)

- NVIDIA DLSS Frame Generation through NVIDIA Streamline (from the DS fork), with Saros's depth
  and motion vectors: the screen shows two or more frames for every frame the game renders.
- Needs an RTX 40 or newer card and NVIDIA's Streamline SDK, which is not shipped here.
- Launcher row "Frame generation", or `run-windows.ps1 -FrameGen <frames>`.
- Menus, movies and loading screens are shown as they are.
- Details: [docs/frame-generation.md](docs/frame-generation.md).

### DLSS upscaling

- **NVIDIA DLSS makes Saros's 4K picture in place of the game's own upscaler.** The game's
  upscaler breaks thin and distant things up into stairs and dots, the more the smaller the 3D
  it starts from; DLSS makes a clean picture of the same 3D.
- Launcher row "Upscaling", or `run-windows.ps1 -Dlss <quality|balanced|performance|ultra>`:
  the 3D at 2560x1440, 2227x1253, 1920x1080 or 1280x720 (the game's own: 1600x900 to
  2304x1296).
- At one spot (RTX 4090): the game as it is 41.7 fps, DLSS performance 41.3, DLSS ultra 46.7,
  balanced 39.5, quality 32.8. At the same 3D size DLSS costs about what the game's upscaler
  does, because the game's own last pass is left out.
- Needs an NVIDIA RTX card and DLSS's model (`nvngx_dlss.dll`) from NVIDIA's Streamline SDK,
  the folder frame generation uses; both work together. Without them the game's upscaler stays.
- Details: [docs/dlss-upscaling.md](docs/dlss-upscaling.md).

### 3D resolution and the game's own Unreal settings

- Unreal Engine titles keep their console variables; the emulator can now set them in the running
  game without patching it: `--unreal-cvar <name>=<number>` or `=x<factor>`.
- "3D resolution" in the launcher (`--unreal-render-scale <percent>`, `run-windows.ps1 -RenderScale`)
  scales the size the game renders its 3D at; the game's upscaler still makes the full picture.
- Saros at one heavy spot: 31 fps as it is, 33 at 75%, 36 at 50%.
- Details: [docs/unreal-settings.md](docs/unreal-settings.md).

### Occlusion queries

- **The game's occlusion queries are answered.** A game asks the graphics card how much of an
  object can be seen and stops drawing what is hidden. The emulator answered "visible" every
  time unless it was started with `--occlusion-queries`. The launcher (Graphics, "Occlusion
  queries") and `run-windows.ps1` now turn that option on; `-OcclusionQueries off` gives the old
  behaviour. The notes of the [Jetsku fork](https://github.com/Jetsku/KytyPS5) pointed here.
- Saros at 50% 3D resolution, at two heavy spots: 41 to 50.6 fps (+24%) in a large hall, with
  1,547 to 619 draws a frame, and 41 to 57 fps (+37%) at another, with 1,540 to 725. Where
  little is hidden the gain is small: 51.5 to 53.7 fps while turning all the way round. With
  frame generation on it is the same gain.
- **Their results no longer stop the thread that feeds the GPU** (after the Jetsku fork). They
  were written at the next label, flush or flip, after a wait for the GPU: one wait of 2 ms a
  frame in Saros. They are now written when the GPU has run them, by the thread that writes
  the labels and ahead of the labels that follow: 50.1 to 51.2 fps in three off/on rounds
  (`occlusion_defer`).
- Checked: the still picture is the same, and captures taken while the camera turned, and
  right after a fast turn, miss nothing. Not judged by eye in motion. That fork turned its own
  queries off by default: they cost half the frame rate in the heaviest levels of another game.

### Smoother frames

- **A frame is shown when it is done, not at the next tick of a 60 Hz clock** (after the
  Demon's Souls fork's `KYTY_VRR`; the Jetsku and the Ghost of Yōtei forks do the same). A
  finished frame used to wait for the emulator's next vblank, so a game that renders 50 frames
  a second showed each one for 16.7 ms or 33.3 ms by turns. It now goes to the screen the
  moment the GPU has finished it, at most once per vblank period; the game's own vblank clock
  is untouched. Saros at a heavy spot with DLSS and frame generation, switched off and on twice
  in one session: the same 50 fps, frame times 16.6 ms (median) and 33.7 ms (95th percentile)
  before, 19.0 and 24.3 ms after; how much a frame's time differs from the one before it
  (90th percentile) 17.2 to 5.4 ms; the 1% low 19–24 to 27–28 fps. `KYTY_FLIP_WHEN_DONE=0` in
  the launcher's Environment field gives the old behaviour.
- **A slow frame every eighth frame at heavy spots is gone.** Every eighth frame Saros's render
  thread reads back 4 to 4.6 MiB the GPU wrote a moment before, from one of five buffers in
  turn. The emulator has a set of "hot" pages for this: memory the game read back once is
  copied out with the GPU work that writes it next, so the game finds it ready. The set forgot
  these pages before their buffer came round again and took only half of each read, so the
  thread fetched the bytes in nine round trips, 10 to 15 ms, and the frame took 30 to 38 ms
  instead of 19: about one and a half hitches a second while standing still. The set now holds
  eight times as many pages and whole reads. Off and on twice in one session: the slowest 1%
  of frames 35.9 ms to 22.0 and 23.2 ms, the longest frame 40 and 76 ms to 24 and 26 ms, the
  1% low 24–27 to 42–44 fps (`hot_pages_max`, `hot_window`).
- **Loading while you move holds the game's threads up less.** While a level streams in, the
  game's loading threads overwrite memory the GPU wrote earlier, thousands of pages in a row,
  and every page was fetched back from video memory by itself first (0.05 to 0.26 ms each, 50
  to 250 ms for a burst, with the GPU thread out of work meanwhile). A thread that walks
  through such pages now gets them a growing window at a time, and a read that runs on
  through a buffer gets larger pieces. A cycle restarted from the pause menu and the walk
  after it, one run each way: 52 to 40 slow frames, 3.3 to 2.7 s lost to them, the game's
  threads held 3.4 to 1.8 s by these fetches (`store_ahead`, `readback_ahead`).
- **Bytes fetched back from video memory no longer land on newer data.** Three parts of the
  emulator write such bytes into the game's memory (the GPU thread's own fetch, a game
  thread's copy of a fetch handed to it, and the background delivery of the pages the game
  reads every frame), and they did not agree on which of them owned a page: two of them
  marked neighbouring pages as delivered that they had no bytes for, and the third wrote its
  bytes without looking whether the page was still waiting for them. A page could so be
  written by the game and then have the GPU's old bytes put over it: a block of a streamed
  texture or mesh with old data in it, or a count that a compute shader then looped on
  without end (the GPU at 100%, the picture frozen, at times ending as a lost graphics
  device with no fault address; one start of about fifteen into a new cycle showed that).
  The race is older than this branch's work on it; the growing windows above made it
  likelier, which is how a review of them found it. Each of the three now delivers and marks
  only the pages it holds bytes for, and only while a page still waits for them. The pages
  ahead of a store are fetched by the storing thread's own copy alone, at most 63 at a time
  (256 before), which gives back part of the gain above: 66 slow frames in the same restart
  and walk.
- **The long frames of a level streaming in are mostly gone.** What a storing thread fetches
  back comes out of video memory through a mapping, and that mapping gives the whole process
  about 85 MB a second however many threads read it (measured in play: 10.5 to 12.3 µs a
  KiB). A block of 4 MiB given to new data was 45 ms of the game's threads standing in their
  stores, with nothing for the GPU thread to draw; when a dozen threads filled one block
  together the frame took 60 to 150 ms. While the graphics card has nothing to do at that
  moment, the GPU thread now brings such a window itself: one copy on the card into ordinary
  memory (about 0.1 ms and 0.3 µs a KiB, a thirtieth of the mapping's price), written to the
  pages that still wait for it, which are then handed to the game. A thread gets eight times
  what it asked for that way, as far as the GPU-written pages go, so the first thread's
  window brings what the others were about to ask for. A player's session of 137 seconds
  played back three times each way (see "Tools for measuring"): frames over 50 ms 15–23
  without it, 3–5 with it; frames in which many threads store at once and the picture
  stands 8–15 to none; time lost to slow frames 1.24–1.60 s to 0.64–0.70 s; the 1% low
  20.6–23.2 to 27.2–28.1 fps; the frame rate the same (`store_ahead_gpu`,
  `store_free_factor`, `store_gpu_min`; `store_ahead_gpu 0` is the state before).
- **Memory the game frees no longer stops the GPU thread until the GPU has finished
  everything** (the Demon's Souls and the Ghost of Yōtei forks found the same). An unmap of
  memory a cache knows waited for all GPU work in flight: 4 to 7 ms each time, in half of the
  ten-second windows of a session played through the game. It now waits only for what would
  still write into that memory (`unmap_precise`; 0 gives the old wait).
- **Threads waiting for a page-tracking lock sleep instead of spinning** (as in those two
  forks). While a level streams in, a dozen of the game's threads wait for these locks at
  once; each burnt a whole CPU for as long as it waited, 35 to 45 ms a thread in one slow
  frame. The same restart and walk, one run each way: 67 to 49 slow frames, a quarter less CPU
  time in them (`lock_sleep`). The thread that feeds the GPU spins on as before: a sleeper
  gets in only once a thread that takes the lock again and again is done, a spinner at the
  first gap, and the frame waits for that one thread (`lock_sleep 2` lets it sleep too).
- Walking the same route round Saros's hub with the build before these changes and after
  them (frame generation and DLSS on, about 40 fps both times): frame times 17 ms (median),
  33.9 ms (95th percentile) and 50 ms (99th) before; 24 ms, 25–30 ms and 26–43 ms after.

### Shaders compiled before play instead of during it

- **Precompile pass** (`run-windows.ps1 -Precompile`, or the launcher's button): builds everything
  known for the game into the driver's cache and exits.
- **The pass runs by itself before the game starts when it has to** (launcher switch "Before
  playing", on by default; `run-windows.ps1 -PrecompileFirst on`). A data folder keeps a stamp
  of its last finished pass: the emulator, the game, the shader lists and the graphics driver
  it was for. A start that finds another stamp, or none, builds first and then starts the
  game. Before this, a game started in a folder whose shaders were not built compiled them
  while it ran: one such session stalled for 9.3 s in its first seven minutes, a second at the
  longest. Closing the pass's window starts the game at once.
- **The whole game's compute shaders**, taken from the game's own files (`--import-shaders`,
  `shader-seeds/`), not only the ones an earlier session met. The precompile pass reads them out
  of an Unreal Engine 5 game by itself the first time it runs (a few seconds for Saros: 37,545
  shaders from six shader libraries), and again after a game update. The test build of
  2026-11-16 could not do this yet: without a seeds file made by hand, its pass built only what
  sessions had recorded and the Saros lists.
- **Pipeline lists** (`pipeline-lists/`): recordings shared without any game code; a list for Saros
  is included. The list of 2026-11-17 has 14,603 compute programs and 977 graphics pipelines
  (12,655 and 610 before): the recordings of every session since, five starts of a new game
  among them. `-ExportPipelineList` now adds a data folder's recordings to the list instead of
  writing them alone.
- **Shader pairs** (`shader-pairs/`): vertex and pixel shaders likely to be drawn together are
  built ahead; a list for Saros is included.
- **At every start**, the recorded programs are translated on helper threads and recorded
  pipelines are built in the background before they are needed.
- **Translated shaders are kept on disk between starts** (`_PipelineCache/<title>.v1.modules`,
  `module_store`). A session translated 12,600 programs into 19,900 modules in the background
  during its first 70 to 80 seconds, 250 s of CPU time, and the frame rate was a fifth lower
  while that ran: half of it was writing the Vulkan shader code, which is the same at every
  start. That half is now read back from a file (664 MiB for Saros, compressed) that the
  first start after an update fills; a second start takes all 19,871 modules from it and
  compiles none. A module is checked when it is stored (read back and compared) and the file
  belongs to one build of the emulator, one graphics device and one set of shader settings.
- **The game's first frame waits until the shaders prepared ahead are ready** (`prepare_hold`:
  the most seconds, 90; `KYTY_PREPARE_HOLD=0` for no wait). With the store filled that is
  about 15 s on an i9-13900K, during which every thread works on them, instead of two minutes
  of play at a lower frame rate: the first three ten-second windows of a played-back session
  41.5, 46.3 and 42.4 fps before, 51.4, 52.7 and 45.9 after.
- **Shader work that runs beside the game stays on the efficiency cores** of a CPU that has
  eight or more of them (`background_ecores`), so it does not take turns with the threads that
  make frames.
- **A faster memory allocator on PCs with 28 GiB of RAM or more** (mimalloc, from the DS fork;
  `KYTY_MIMALLOC=0` or `1` decides instead). Shader translation spends about a third of its
  time allocating, and its threads waited for each other at the one heap: the wait before the
  first frame went from 22.0 to 15.1 s, and the thread that feeds the GPU works 0.7 ms a frame
  less.
- **"Never wait for a new graphics shader"** (`--skip-pending-draws`): a graphics shader met for
  the first time is built in the background and its draw is left out until it is ready, instead
  of the game stopping.
- The driver's own shader disk cache is kept in the game's data folder.
- The pass's summary now also says how much earlier passes already built, so a short re-check no
  longer looks like a lost cache.
- **The pass builds on every CPU but two.** It used half of the threads and never more than
  eight. Nearly all of its time is the graphics driver's shader compiler (94 to 98% of a build
  thread with an NVIDIA card), so the gain is what the extra threads are worth: on an i9-13900K
  the same first 3,000 builds take 118 s instead of 208 s, and the first phase (the recorded,
  listed and compute shaders, 13,800 builds) 15 minutes instead of about 27. On a PC with
  little memory it takes fewer threads; `-PrecompileThreads <n>` names a count.
- **The progress line says how far the pass is**: the total, the share done and about how long
  is left at the pace of the last two minutes.
- **A pass can be stopped and continued.** Closing the emulator's window ends it; what it built
  is saved then, and every five minutes while it runs, and the next pass goes on from there.
  Before, the save after a stop could fail and lose everything the pass had built, a pass saved
  nothing until its end, and stopping twice during the same long compile shut that shader out
  of later passes.
- **"Also build likely shader pairs"** in the launcher (on by default; `-ShaderPairs none`):
  off, a first pass leaves out the shader pairs, about 23,000 of the 37,000 pipelines of Saros.
- Details: [docs/shader-precompile.md](docs/shader-precompile.md).

### Crashes and memory

- **The game no longer reads old bytes from memory the GPU has written.** A page the game reads
  back every frame stays readable while the GPU writes it, and its new bytes are copied back
  when the submission completes. Six paths let the game read the bytes of before the work
  after it had been told the work was done, or put older bytes over newer ones; in Saros a
  worker thread took an index from such a record and crashed the game (an access violation in
  the game's code, in 3 of 8 test runs of a level within a minute):
  - the mark that keeps a page readable stayed behind on neighbouring pages (since the
    publication fix of 2026-11-24 a copy-back cleans only the pages it holds bytes for), and a
    later GPU write of such a page left it readable with its old bytes for good;
  - the game's own "done" markers for a submission (its labels, the flip, a suspend point) were
    queued before that submission's copy-back and so were delivered first. The copy-back now
    runs before anything else queued for its submission (`publish_first 0` is the old order);
  - a readback that took over such a page's bytes delivered them behind the labels while the
    page stayed readable; and a page written twice in a row stayed readable for a moment after
    the first write's bytes had been given up;
  - a copy-back made its pages readable before it wrote their bytes, so a thread that read at
    that moment got the old ones (the bytes are written first now); and a page dropped from
    the full set of such pages let a copy-back already on its way put its bytes over a GPU
    write recorded after it. A review found these two; neither was seen to happen.
- **Eight fixes of the Saros fork's `saros-v0.1.8` to `saros-v0.1.12` are taken over** (its author
  in turn took our lazy buffer synchronisation, the white flash fix and the readback work). Each
  is its own commit with his text:
  - *Lost graphics device*: a block-compressed texture could be given more mip levels than its
    host image holds, and uploads of the extra ones wrote past the image's memory; and a draw
    could keep an image that another binding of the same draw had just replaced. Both end in a
    lost device once that memory is freed.
  - *Wrong textures after a respawn* ("the rainbow outfit"): an image created again over memory
    that had just been written back was uploaded from a stale buffer.
  - *Objects missing with occlusion queries on*: a draw the emulator leaves out inside one of
    the game's queries (its pipeline is still being built, its shader cannot run, it is
    replaced by a copy) counted as hidden, so the game could cull what its draw shows. It now
    counts as visible. Unlike the Saros fork, such a draw does not wait for its pipeline here:
    it is left out as outside a query, so the frame does not stop for the build.
  - *12 GB cards*: when the video memory budget fell during a load and came back seconds later,
    the buffer collector stayed at the lower mark for the rest of the load and evicted buffers
    shaders still read.
  - *A crash*, `failed to read color metadata backing`, after the game unmapped the memory a
    texture's metadata was in.
  - *Reports*: a crash of the game now also lists the memory around its registers, the last
    faults the emulator resolved on that thread and where every other game thread was; and a
    scan reports up to 8,191 buffer accesses where it reported 1,023.
  Not taken: what follows his move to a newer base of the main project (the pipeline record
  formats 8 and 9), his launcher's settings, the background rebuild of a stale driver cache,
  and the pieces this branch has in its own form (the device fault record, the commands of a
  lost submission, the equal-address volume fix). Tried and set aside: his selection feedback,
  which stops keeping images alive for textures a shader's table lists and never selects
  (several GiB of video memory while walking in Shattered Rise, his number). It builds here,
  but its GPU test fails on this branch's base (the test's shader samples the first texture
  of its table whatever the key, also with his new code switched off), so it stays on the
  side branch `wip/selection-feedback` until that is understood.
- **The Saros fork's fixes up to `saros-v0.1.7` are merged**: the emulator follows the driver's
  current video memory budget and frees memory at once when it is over the limit, a lost
  graphics device on the lift ride into Shattered Rise and in similar spots is fixed (an older
  copy of a picture was written over newer game data), and so is a freeze when video memory
  was nearly full. A failed wait for the GPU now names the Vulkan result and the memory heaps
  in the log, and the log warns while video memory is over its budget.
- **The shader preparation gives way where memory is short.** Translating the compute shaders
  of the whole game ahead of their first use holds 6.6 GB of RAM in a Saros session (23.1 GB
  against 16.5 GB at one spot): it is done on a PC with more than 40 GB of RAM, and left to
  the first use (about 13 ms each) on one with less; `KYTY_PREPARE_CREATED=1` or `=0` decides
  it by hand. The pipelines a session keeps ready for their first use take about 0.7 GB of
  video memory, and the background builds at the start of a session leave 0.9 to 2.6 GB
  behind in the driver: when video memory comes near its limit the kept pipelines are
  released and the builds not started yet are left undone.
- **The table that tells shaders where a buffer lives takes the video memory it needs** (after
  the DS fork). It has an entry for every 16 KiB of the game's address space and was made whole:
  768 MiB of video memory on every card, nearly all of it zeros. It now starts at 64 MiB and
  grows to the highest buffer the game registers, 336 MiB in Saros: video memory in use at a
  test spot went from 8.1 to 7.7 GB. `KYTY_BDA_TABLE_GIB=0` keeps it whole.
- **A card short of video memory does not build pipelines beside the game.** Those builds keep
  about 1.6 GB of video memory at a heavy spot (the pipelines held ready, and what the driver
  keeps of each pipeline it made), and giving them up later does not give the memory back. With
  a video memory budget under 13 GiB (cards of 12 GB and less) a session leaves them out: what a
  precompile pass built comes from the driver cache at its first use, anything else is compiled
  then. `KYTY_SESSION_PRECOMPILE=1` builds them on any card, `=0` on none.
- **A buffer is made in video memory even when the budget is used up**, and in system memory
  only when the card has none left. Before, the allocator put it in system memory without a
  word once the budget was reached, where the GPU reads it over the bus at every draw until the
  game is restarted (seen on the DS fork). The log warns when it happens all the same.
- The allocator takes video memory in blocks of 64 MiB instead of 256 MiB on cards with less
  than 13 GiB (from the DS fork, where it gave 2 to 6% on a card of 8 GB; not measured here).
- **A session no longer writes the shader cache again when nothing was added to it.** After a
  precompile pass every session wrote the whole file twice (2 GB for Saros, and as much memory
  for the copy while it did).
- **The log names the PC's memory**: RAM, free RAM and what can still be committed, at the
  start and again with a fatal error. Running out of memory shows as many things (a lost
  graphics device, a crash inside the game); a report can now be checked for it.
- **Much RAM is not much memory to spare, and the emulator now looks at the second.** A
  session here lost the graphics device three minutes into play on a PC with 64 GB: it had
  started with 18.5 GB that Windows could still give out (a web browser alone held 36 GB in 192
  processes) and stopped holding 23.6 GB with 0.6 GB left. Three things follow. The shader
  preparation that holds 6.6 GB steps back when less than 28 GB can be committed at the start,
  not only on a PC with little RAM (so that session would have held about 17 GB). A start with
  less than 24 GB to spare says so (`WARNING: Windows can give out only ...`) in the console
  and the launcher's log. And a fatal error with less than 2 GB left says that Windows was out
  of memory when it happened, the likely reason.
- **A fatal error where the game reuses the address of a 3D render target is fixed** (from a
  player's report, `unresolvable equal-address image overlap`). Saros places a deeper 3D render
  target of another width where a smaller one was, with the same row length; the emulator knew
  the deeper one only when the width was the same. The larger image now takes over what the
  smaller one holds. The message of the cases still not handled names both images' sizes.
- **A lost graphics device leaves a record.** With the driver's `VK_EXT_device_fault` the log,
  and the launcher's log, say how full video memory was and what the driver recorded: the
  address the GPU faulted on, or none (the GPU stopped answering and Windows reset it).
  `KYTY_DEVICE_FAULT=0` leaves the extension off. An address is looked up among the emulator's
  buffers, the live ones and the last 65,536 destroyed, and the record says whether an
  upscaler's commands were in the lost submission. Two records so far: a player's (a write
  through an invalid address, hours into a session) and one here (a read through an address of
  no buffer, during the load into a save, with DLSS upscaling on).
- **The record also says what the last twelve submissions held**: how many guest dispatches and
  the shaders of the last six, render pass instances, buffer uploads, and whether texture
  uploads, downloads or the upscaler were in them. The failing call names the submission the
  GPU did not finish, so the two together say what it was doing. The loss here came back in a
  loop of test starts with DLSS upscaling on, twice in 62 and once before the loop, always a
  read at about the same low address that is in none of the emulator's buffers, during the
  load or at the arrival in the level; none of those starts had this record yet, and the cause
  is not known.
- **The record now lists every submission the GPU had not finished**, not the newest twelve. A
  GPU that stops answering stopped in the first of them, and the GPU thread goes on submitting
  for a while before it has to wait: in the first such record here (no fault address, ten
  seconds after the arrival in a new cycle, frame generation on) the GPU had finished
  submission 45253 and the twelve listed began at 45257. The last 256 submissions are kept,
  and the record is written once where several threads fail on the same loss (it was written
  twice, the lines of both mixed).
- **A start that dies inside a driver says where.** A player's log of a start with frame
  generation ended in the middle of the Vulkan device's creation, without an error: the
  process had crashed inside a library that is not the emulator's, before the emulator's crash
  report is set up, and the file's last 4 KiB were never written. Now a log begins with the
  build, says `Creating the Vulkan device` (through NVIDIA Streamline, with DLSS upscaling
  asked for) before the call, is written out at least once a second and at every message of
  Streamline, and an exception nobody handles is named from the process' first moment:
  `--- Crash ---` with its code and the module it happened in.
- **The crash note names the callers.** The same player's next start, with frame generation and
  DLSS upscaling, ended with that note: an access violation inside the Vulkan loader
  (`vulkan-1.dll`, a read through an invalid pointer) while the device was made. In that loader
  version the address is the check of a device handle (`loader_get_icd_and_device`): the loader
  was handed a device that is not one it made, before the emulator had the new device back. The
  emulator's request and every line of Streamline up to there are the same as in a start that
  works here, so the handle came from what runs inside the call on that PC: NVIDIA's runtime
  asking for its functions, or another program's Vulkan layer (that PC has the layers of Epic's
  overlay and of RTSS, which this PC has not). One address cannot say whose call it was. The
  note now also holds the registers, the bytes of the failing
  instruction, the callers up the stack by module and offset, and the modules in the process
  that are not Windows' own (overlays and hooks of other programs show there).
  `KYTY_CRASH_TEST=1` ends a start with a crash made on purpose, to check the note on any PC.
  The log also says which Vulkan extensions NVIDIA's NGX asked for.
- **The graphics driver gets 64 GB of address space where it had 16.** With the layers left out
  (next entry), that player's start no longer crashed: the log now says what was underneath,
  `Could not create the Vulkan device ... ErrorInitializationFailed`. The NVIDIA driver makes
  the device that DLSS and frame generation ask for only when it finds address space below
  1 TB, and the emulator, which reserves the game's address space up to 1008 GB, left it the
  16 GB above that. Measured on an RTX 4090: the driver reserves two blocks there, each as
  large as it finds. Of 16 GB it gets one block of 8.5 GB and no second, of 64 GB 34 + 17 GB.
  One block is enough for that card; an RTX 5070 on a newer driver was refused. With DLSS or
  frame generation asked for, the game's address space now ends at 960 GB (the game maps
  nothing above 832 GB) and the 64 GB above it are kept free for the driver until the device
  is made. `KYTY_DRIVER_SPACE_GB=<16 to 128>` chooses another size. The log says how much was
  free before the call and how much the call took (`Address space below 1 TB ...`), and the
  error message of a refused device carries the numbers. Not shown yet on the card that was
  refused.
- **With frame generation, other programs' Vulkan layers are left out.** The overlays of Steam,
  Epic, GOG Galaxy and RTSS (MSI Afterburner) and OBS's game capture each put a layer into every
  game, and with frame generation they run between NVIDIA's runtime and the driver. In the
  start above the driver had refused the device, and the crash in the loader is what a layer
  going on with a device that was never made looks like: with the layers left out the same
  start ends with the driver's refusal in plain words. Leaving them out is the
  loader's own switch (`VK_LOADER_LAYERS_DISABLE`, with the graphics vendors' own layers kept),
  set by the emulator when frame generation is on and the environment names no filter itself.
  The log says `Vulkan layers left out` when it applies. What it costs: those overlays and
  OBS's game capture do not show in a session with frame generation (OBS's window capture
  uses no layer). `KYTY_VULKAN_LAYERS=1` keeps the layers; `=0` leaves them out without frame
  generation too.
- **The emulator's log is written by default.** The launchers' Log setting and
  `run-windows.ps1` without `-Log` now choose the file (`logs\<time>.log` in the data folder),
  so a report of a crash has the log without a second try. Measured at one spot of Saros: no
  frame rate lost (56 fps with the log, 53 to 54 without), but the start into the save took
  about 7 s longer (42 and 47 s against 35 and 39), and one such start writes some 25 MB. The
  run script therefore keeps the five newest of these files and removes older ones. Log: Off
  in the launcher (`-Log off`) is as before; a settings file that says `off` keeps saying so.
- With 12 GB of video memory or 16 GB of RAM Saros is still at the edge; a lower 3D resolution
  and no frame generation leave more room.

### Smaller things

- **A white flash every ten seconds or so is gone.** For one frame the scene lost its
  ray-traced shadowing and showed raw, speckled, over-bright lighting, every 608 frames. A
  shader that walks the game's ray tracing structure read "no node here" from a page of it
  while the buffer cache was watching that page (it does so to learn whether the page is still
  read), so every ray hit nothing for the frame or two the watch lasted. The watch is as old as
  the Saros fork; this branch made it regular. Every build of this branch had the flash.
- `KYTY_HOST_GAMEPADS=0` makes the emulator ignore the PC's gamepads.
- A fatal error ends the process at once instead of hanging in library shutdown.
- The emulator opts out of Windows power throttling.

## Performance

At a heavy spot of Saros (about 1,560 draws, 1,120 indirect draws and 830 dispatches a frame, no
frame generation) the branch went from 29.7 fps to about 35 fps at the game's own 3D resolution
during this work, and runs at about 39 to 40 fps at 50% 3D resolution. At a lighter spot the
unmodified `saros` branch gave 37 fps and the first batch of changes below 47.9.

### The thread that feeds the GPU

- **Vulkan commands are recorded on a thread of their own**: 38.0 to 40.9 fps.
- **Buffers are synchronized only where shaders read them through the page table**: 40.4 to
  47.9 fps at the lighter spot, the largest single step. Guest write faults fell from about
  3,700 to 840 a frame.
- **Draws that bind the same small constant buffer share one copy of it** (idea from the DS
  fork): copies 10,900 to 2,600 a frame, 2.6 ms less CPU a frame, about +2.3 fps at 50%.
- **Saros's main texture table stays resident on the GPU**: the table has about 200,000 entries
  and 2,100 textures, and was rebuilt at every draw that reads it. About +1 fps at 100% and
  +3.4 fps at 50%.
- **The look-ahead that prepares resource tables on helper threads plans again where the command
  stream leaves what it scanned** (command lists the game writes on the GPU): table walks left to
  the GPU thread fell from 226 to 53 a frame, 34.1 to 35.1 fps.
- **No barrier between storage writes that draws repeat** (from the DS fork): 30.2 to 31.1 fps.
- **A read that no pending label can touch skips the label list's lock**: about 1 ms less CPU a
  frame.
- **Address ranges are kept in blocks of sorted intervals instead of a tree** (from the DS fork):
  the sets of GPU-written, pending and mapped ranges took 3.3% of the GPU thread at the heavy
  spot and take 1.1%; 24.3 to 23.6–24.0 ms of that thread a frame at 50%. The frame rate moved
  within what two runs differ by (40.4–40.7 before, 40.1–41.6 after).
- **A GPU-driven draw or dispatch that does nothing is left out when that is known without a
  wait** (`skip_empty_indirect`, off by default, see `fault_ahead` below; the fork of Jetsku
  skips empty indirect dispatches the same way). It rarely is in Saros, and that is the finding: at the heavy spot 170 of 310 GPU-driven
  draws a frame draw no instance and 180 of 420 indirect dispatches run no workgroup, each
  with the full preparation of a draw or dispatch, but their arguments are written by a culling
  pass recorded just before them and are still with the GPU when the packet comes. Two or
  three draws a frame are found without waiting. Waiting for the arguments finds all of them
  (draws 604 to 433 a frame, dispatches 812 to 673) and costs more than it saves: seven more
  waits a frame, 49.5 to 45.0 fps (`skip_empty_indirect 2`, for measuring).
- **An option that guesses which GPU-driven draws are empty** (`predict_empty_draws`, off by
  default; `KYTY_PREDICT_EMPTY_DRAWS=1` in the launcher's Environment field turns it on). It is
  not exact: a draw whose record the GPU still holds is left out when the same draw (same
  shaders, index buffer, targets, viewport and place in the frame) had nothing to draw the last
  four times it was checked and nothing in the last 90 frames. Every such draw's record is
  still read each frame, by a copy at the draw's place in the command stream that arrives
  without a wait, so a draw that gets something to draw is recorded again a frame or two
  later. At the heavy spot about 125 of the 170 empty draws a frame are left out, draws 604
  to 482, the render thread's own work 16.1 to 14.4 ms a frame; about one draw in ten frames
  is left out wrongly, standing, turning or walking, and stills with it on and off do not
  differ. Most of the saved time comes back as waiting for the GPU: 50.0 to 51.3 fps in one
  session, 50.5 to 51.2 in another.
- **A page that stays readable through a GPU write takes one protection change, not two.** The
  pages the game reads back every frame stay readable while the GPU writes them. The write
  was marked first, which took the page from the game, and the page was given back right
  after: two requests to Windows that cancel, on the GPU thread, under the lock the game's
  write faults wait for; with the set of 16,384 such pages about eight page changes for every
  page delivered, every frame. Both states are now set in one update. A player's session
  played back, one run each way: pages changed by the GPU thread 19,000–36,000 to
  3,800–6,200 a frame, that thread's CPU time 17.8–18.3 to 16.7 ms a frame, 45.4–47.7 to
  49.8 fps (`keep_marked`).
- **The look for things to release before each draw no longer sleeps.** Buffers and images
  given back are released once the GPU has finished with them and their submission's results
  have been delivered. The thread asked before every draw and dispatch, some 1,500 times a
  frame: a question to the driver and a lock each time, and where something was due, a sleep
  until the delivering thread had caught up, 0.42 ms a frame. It now asks only when the oldest
  entry can be due, without the lock, and leaves an entry for the next draw instead of
  sleeping (`pop_no_wait`).
- **The pages the game reads back are checked and protected by runs, not one at a time.**
  What a download delivered was taken off the list of bytes the GPU still holds with a
  question and a subtraction for each 4 KiB page, 6,700 pages a frame: adjacent pages are now
  one question and one subtraction, 470 a frame (`drain_runs`). And the pages a GPU write
  makes unreadable took a locked question and a protection request each: all pages of a write
  are now handled with one request per region, 933 to 591 requests a frame
  (`unreadable_merged`). With the item above, switched off and on six times in one played-back
  session: 15.13 to 14.64 ms of the thread's CPU time a frame.
- **A dispatch that would wait for the GPU is put aside while the game's other queue has
  work** (`suspend_readback`). Saros runs its GPU-driven culling in a compute queue: a pass
  writes a count, the next dispatch needs it on the host, and so on, four times a frame right
  after the frame's heaviest GPU work. The thread stopped at the first until the GPU had
  caught up (up to 4.5 ms of a 19 ms frame) while the game's graphics queue had commands ready.
  Such a dispatch is now suspended before it records anything, as a wait for a label is; its
  bytes are sent for without a wait, the thread feeds the other queue, and the dispatch runs
  again when the GPU has finished the submission the bytes come with. With nothing else to
  run it waits as before, and it is never put aside twice for the same bytes. Waits 5.2 to 1.3
  a frame, the thread's frame 18.5 to 17.7 ms, 53.7 to 55.6 fps (two whole played-back
  sessions; the same in off/on windows inside one).
- **A texture that changes takes its places in the resident texture table without a search of
  the table.** Saros uses one block of memory as its depth buffer and, in a compute pass, as a
  colour image; the image is made anew at each change of use, and the table's two entries for
  it changed every frame. Finding the table places that pointed at them meant looking at all
  198,000 of them, 0.4 ms a frame. Each texture now keeps a list of its places: 385 to 35 µs a
  frame, 55.2 to 56.2 fps (`heap_slot_lists`; `heap_verify 1` checks the lists against the
  full search in every frame, and a minute of play found no difference).
- **A protection change inside the mapping a thread asked about last skips the search for it**
  among some thousands of mappings (from the DS fork).
- **Command buffers are no longer ended early for colour metadata nobody reads on the CPU.** A
  compute pass that writes a render target's fast-clear data ended the command buffer at once,
  so that a read of that data on the CPU would wait for little. Saros's is always checked on
  the GPU, never on the CPU: 340 to 450 submissions a second were for nothing, 94–97
  submissions a frame are now 89–93. A range is treated that way only once it has been read
  on the CPU. (With it, upstream's change that compares such data a block at a time without
  allocating.)
- The GPU thread runs above the game's own threads.

### Readbacks (the game reading what the GPU wrote)

- **A game thread finishes its own readback** instead of waiting for the GPU thread: 29.6 to
  31.3 fps, and the slowest 5% of frames from 49.6 to 33.8 ms.
- **Only the pages a store or a table walk asks for are read back**: 38.8 to 40.3 fps.
- **Readbacks over 64 KiB go through the download buffer**: 40.7 to 41.6 fps, worst frame 66 to
  34 ms.
- In-place readbacks copy with streaming loads.
- **A value only the shader takes is copied to it on the GPU instead of read back** (the DS
  fork's direction for GPU-written draw arguments, here for the values a resource walk loads).
  Saros chains small compute passes, each loading a few bytes the one before wrote, and the
  thread that feeds the GPU stopped for every such load: 8 waits a frame at the heavy spot.
  The walker now knows which loaded values nothing on the host depends on; for those a
  compute dispatch copies the bytes into its own data on the GPU. Waits 6.9 to 5.0 a frame,
  35.0 to 35.6 fps in four off/on rounds on a busy PC (`gpu_scalars`). The waits that remain
  are values the host needs itself: two counts that go into a buffer's address, the thread
  counts of two indirect dispatches, and a command list the game builds on the GPU.

### The game's stores into tracked memory

- **A store that walks through tracked pages opens them a window at a time** (after the
  fault-ahead of the Jetsku fork). Saros took about 1,700 write faults a frame on pages the
  buffer cache had write-protected, three in four of them beside another one: game threads
  filling a buffer page after page, each page an exception and a protection change, and
  another protection change when the page is uploaded. A thread that faults on the page after
  its last fault now gets the following pages opened with it, 2, 4, 8 and then 16 at a time; a
  single fault opens nothing, so scattered stores cost no extra uploads. The window stops at a
  page the GPU owns and at one an image watches too. At the heavy spot with DLSS at 1440p:
  faults 1,720 to 585 a frame, protection changes 2,730 to 1,785, the emulator's CPU time 111
  to 104 ms a frame, buffer uploads +3%. The frame rate there is held by the thread that feeds
  the GPU and moved from 51.8 to 52.2 fps; the picture is the same (`fault_ahead`: the most
  pages a fault opens, 0 for none). **Off by default** (`KYTY_FAULT_AHEAD=16` turns it on): in
  the loop of test starts above, 2 of 26 lost the graphics device with this and
  `skip_empty_indirect` on against 0 of 26 with one or both off. That neither convicts nor
  clears them, the same loss was seen once before either existed, and neither makes a frame
  faster, so they wait for the cause.
- **A store of the command stream over bytes the GPU wrote is not fetched back first**
  (`write_data_gpu`, `write_data_gpu_words`, `store_both`): its words are copied into the
  buffer in the order of the commands, and written into the game's memory as well, where they
  stop counting as bytes someone has to wait for. Saros writes one word that way about every
  second frame and a record of 56 words every frame; for the record the thread that feeds the
  GPU stopped 0.85 times a frame until the GPU had caught up. Those stops are gone, and the
  frame rate is the same (19.60 against 19.64 ms of that thread a frame): the next read of
  GPU-written bytes, a third of a millisecond later, now waits for the same work. The finding
  is in "Not done" below: the waits of a frame come one after the other, and the first pays
  for all.

### Textures

- **A rendered image is no longer uploaded again from stale memory when a small unrelated buffer
  is written at its address** (Unreal reuses one block of memory for several resources in a
  frame): image uploads 320 to 130 MiB a frame, about +1 fps at 100%.

### Shader translation

- **Compute programs whose 64 lanes do not depend on each other run one lane per invocation**
  (from the DS fork): 357 of Saros's 563 such programs qualify; GPU time -1.7%.
- **One module per compute shader serves whatever the game binds**: buffer formats, strides,
  texture channel orders and level counts are read at run time, which is what lets the whole
  game's compute shaders be built before play.

### Switches for comparing

Most of the changes above can be turned off while the game runs, through the live channel below:
`vulkan_recording`, `bda_lazy`, `stream_share`, `heap_by_content`, `lookahead_replan`,
`guest_image_writes`, `image_buffer_writes`, `guest_copy`, `readback_narrow`, `gpu_scalars`,
`fault_ahead`, `game_upscaler`, `game_upscaler_skip`, `flip_when_done`, `hot_window`,
`hot_pages_max`, `store_ahead`, `store_ahead_gpu`, `store_free_factor`, `store_gpu_min`,
`readback_ahead`, `unmap_precise`, `lock_sleep`, `publish_first`, `keep_marked`,
`write_data_gpu`, `write_data_gpu_words`, `store_both`, `pop_no_wait`, `drain_runs`,
`unreadable_merged`, `module_store`, `prepare_hold`, `background_ecores`,
`suspend_readback`, `heap_slot_lists`. Each also reads an
environment variable `KYTY_<NAME>` at
start. `present_crop` 1 to 9 shows a part of the picture pixel for pixel in the window.

## Tools for measuring

- **Live channel** (`KYTY_LIVE_FILE=<file>`): commands written to a file while the game runs.
  `measure <seconds>` reports fps, frame times, GPU time and per-frame counters; `prof` samples a
  thread and can look inside one function; `set <switch> <value>` flips a switch; `mem` searches,
  reads and writes the game's memory; `pad` and `stick` send controller input. `measure`
  switches the performance panel's counters on for its window; `fps <seconds>` reports the
  frame rate alone and leaves them as they are (off and on gave the same rate on Saros).
- **A performance log of a whole play session** (`KYTY_PERF_LOG=<seconds>`, for example
  `KYTY_PERF_LOG=10` in the launcher's Environment field): every such window of play writes one
  `Perf <time of day> ...` line with the numbers of `measure` into the log, so a session played
  through any part of the game says afterwards where it was slow and whether the CPU or the GPU
  was the limit there. Windows without a frame (loading) are left out.
- **A hitch log** (`KYTY_HITCH_LOG_MS=<milliseconds>`, for example `KYTY_HITCH_LOG_MS=30` in
  the launcher's Environment field; the Demon's Souls, Jetsku and Ghost of Yōtei forks each
  have such a log). `measure` and the performance log average over seconds, in which one frame
  of 80 ms among 500 does not show. With this, every frame that stayed on screen at least that
  long, and half as long again as the frames before it, writes one `Hitch <time of day> ...`
  line: the frame's time beside the usual one, then, as `this(usual)`, the GPU thread's busy
  time, its waits by reason, and whatever else lies far above a usual frame: shaders and
  pipelines built, uploads, evictions, the CPU time of every emulator thread and of every one
  of the game's threads by name. `| held ...` lists what the emulator made the game's threads
  wait for in that frame (fetching GPU-written memory back for a read or a store, an unmap, a
  slow write fault), by thread, with the longest wait and its address. It found the two
  causes above: frames in which the GPU thread sits idle because a game thread is held.
  `measure` also reports `step_ms`, how much a frame's time differs from the one before it,
  and the pages the hot set delivers a frame (`async_pages`).
- **Both logs run without the performance panel.** The panel was the counters' own switch, so
  either log put it on the screen for the whole session and F2 could not hide it. Now a log
  holds the counters on by itself; F2 shows and hides the panel as before.
- **A mark in the log** (the F9 key, or `mark` on the live channel): `Mark <time of day> #<n>`
  and, two seconds later, `MarkPerf #<n>` with the numbers of `measure` for those two seconds.
  For saying afterwards where in a long log something was seen on the screen (a wrong picture
  has no line of its own). A key mapping that uses F9 keeps it.
- **A play session can be played back along its path** (`replay <log> <from> [seconds]
  [oncamera]` on the live channel), as a benchmark of real play. A session's log already held
  every gamepad event with its time; it now also holds the camera of every frame (`Camera:`
  lines: position and direction, read from the game's view constants as the upscaler reads
  them; `path_log 0` leaves them out). Played back by the clock alone the game leaves the
  session's course within seconds, because a stick sampled one frame later turns the camera a
  few degrees further. So the replay corrects the sticks towards the recorded path: it turns the
  camera to where it looked and walks the player to where they stood, slows or speeds the
  session's clock a little for a player who is behind or ahead, and, where the player has left
  the path and stands on another part of it, goes on from that part. `Replay track:` lines say
  every second how far off the path it is, and the end line until when it kept to it. On a test
  path of 76 seconds with a drop off a ledge it stayed within 1 to 35 units and 1.5 degrees
  while running and on the path for 66 to 76 seconds; a player's session of 137 seconds
  (running, jumps, a fight) was followed to its end, 20 units and 0.6 degrees off on average,
  with the frame rate of every ten seconds within a few percent of the session's own. What it
  cannot repeat is a fight: the enemies do not do the same thing twice. The camera is read
  where the emulator's upscaler or frame generation reads it, so one of the two has to be on.
- **What was done to the memory a crash read** (`mem_trace 1`, `KYTY_MEM_TRACE=1`): a ring keeps
  the emulator's last 131,072 actions on memory the GPU writes (a GPU write recorded, a page
  left readable or taken from the game, bytes copied back and by which way, faults, labels and
  flips delivered), and a fatal report lists them for the memory each register points to, with
  their times against the fault. For a game thread that crashed on a value it read too early.
- **What a large buffer costs to make and to free**: a buffer of 64 MiB or more, or one that
  takes 5 ms or more, writes a `BufferCache:` line when it is made (its memory, the buffers
  copied into it, the rest) and when its memory is given back. Saros's heap grows 16 MiB at a
  time, and each time the buffer over it (350 MiB and more) is made anew.
- **What the mapping of video memory gives** (`read_mapped_kb`, `read_mapped_us` in the
  performance log): 80 to 93 MB a second in a played session, about a gigabyte in two and a
  half minutes; and `store_gpu_kb`, `store_gpu_us` for the same bytes brought by a copy on
  the GPU. Counters under ten a frame are printed with three decimals, so what happens a few
  times in a window can be counted from its line.
- **What the thread that feeds the GPU waits for, by cause.** Counters in the performance log:
  `sync_passes`, `sync_gpu_us`, `sync_completion_us` (its waits and their two parts),
  `sync_read_own`/`sync_store_own`/`sync_read_guest`/`sync_store_guest` (who asked),
  `settle_*` (the command stream's own reads, by kind), `walk_readbacks`,
  `write_data_fetched`, `handoff_*` (why a game thread's readback was not handed to it). Live
  switches that write a line per event: `sync_log` (what was read, how long the wait was, how
  far the GPU was behind, which of the game's queues runs and whether another could),
  `walk_read_log` (the shader and the bytes), `settle_log`. At start,
  `KYTY_WALK_EXPLAIN=<shader hash,...>` prints what a shader's resource table walk computes
  from what it loads, and `KYTY_WATCH_WRITE=<address,...>` names the shaders that bind an
  address for writing and how much they bind around it. The performance log's lines end with
  the emulator's memory (`mem_mb working`, `private`, `free`).
- `readback_log` (live switch, for a few seconds) writes a line for every readback a game
  thread's read asks for and for every store of the GPU thread into GPU-written pages;
  `stale_readable 0` with the counter `early_reads` counts reads that arrive before the
  delivered bytes are in place.
- **The GPU's time by shader** (live switch `gpu_prof`; the idea of the Jetsku fork's operation
  profiler). While it is on, a timestamp is recorded before and after every guest dispatch and
  render pass instance, and turning it off writes `GpuProf` lines into the log: the GPU's time a
  frame for each compute shader (by its hash, with how often it ran), for draws, buffer
  uploads, texture uploads, downloads and the upscaler. The timestamps make the commands run
  one after the other and cost about 3 µs each, so the frame rate drops while it runs; the
  shares are what it is for. At the heavy spot of Saros with DLSS at 1440p it says: compute
  shaders about 60% of the GPU's time, spread over 350 of them with the largest at 5%, the
  upscaler 8%, draws 7%, uploads and downloads together about 11%. It also settled a guess:
  leaving out the 440 compute-to-all barriers a frame (a test build) changed neither the GPU's
  time nor the frame rate, so barriers are not what the GPU waits on.
- **A card with less video memory, for tests on a bigger one** (`KYTY_SIMULATE_VRAM_MB=<n>`, from
  the DS fork): the rest of the card's memory is taken at start and kept, so the driver and
  Windows treat the game as on a card of n MiB.
- **An upload worker** for buffer synchronization exists behind `async_upload` and is off: it did
  not pay on Saros.
- `build-windows.cmd` can build with profiling symbols and with a locally trained PGO profile.

## Not done, and known limits

- Table mode of the DS fork is not ported: as written it takes only draws without bound vertex
  buffers, which is 3% of Saros's draws. Its ideas that carry over are in the list above.
- **The thread that feeds the GPU still waits 5 times a frame, 5.5 ms of a 19 ms frame**, for
  bytes the GPU writes and the host needs itself, and what they are is now known (a played-back
  session, 1440p with DLSS): the waits come one after the other, right after the frame's
  heaviest GPU work, so the first pays for all of that work and the others are short. In
  order: a compute shader takes the record count of a buffer from a number a culling pass
  just wrote (4.5 ms: the GPU is 25 submissions behind there); an indirect dispatch reads its
  thread counts (0.2 ms); a shader reads a header word that never changes but lies in a block
  of memory its writers bind whole (0.1 ms); a second indirect dispatch (0.2 ms); and the
  graphics queue calls a 35 KiB command list the GPU wrote (0.3 ms). Removing one only moves
  the wait to the next, measured twice. The first four run in one of the game's compute queues
  while its graphics queue has commands that could run, and are now put aside for that queue
  (see "A dispatch that would wait" above). What is left: the call of the GPU-written command
  list, which now pays 1.1 ms a frame (the GPU is nine submissions behind there and no other
  queue can run), and readbacks for the game's own threads that this thread still waits for
  (0.3 ms a frame). The picture with the queues interleaved this way has been compared by
  counters only, not yet by eye.
- Time saved before those waits comes back as longer waits: an upload worker for the per-draw
  uploads (the DS fork's scope) cut that thread's copies and page protections by 5% of its
  time and changed no frame rate for that reason; it is not in.
- A summary per 4 MiB tracking region, so that a buffer bound for reading over hundreds of MiB
  skips the regions the game did not write, was built and reviewed: it answered 35,600 of the
  40,400 region questions a frame with one load, and saved nothing that could be measured
  (the question it replaces takes a few nanoseconds). Not in.
- The DS fork's merged page protections are not ported: they join protections of adjacent
  pages that follow each other, of which Demon's Souls has about 110 a frame and Saros one.
- Cards of 12 GB and less are not confirmed. With the rest of a 24 GB card taken away
  (`KYTY_SIMULATE_VRAM_MB=12288`, on a desktop whose other programs held 5 GB of it) Saros ran
  at 20 fps at 100% 3D resolution and at 52 fps at 50%, where the whole card gives 53 and 60:
  Windows kept 1.3 to 2 GB of the game's video memory in system memory.
- Three reports of a lost graphics device are not explained: one on the first test build, two on
  the build of 2026-11-17c (an RTX 4070 with frame generation, less than a minute into a
  session, with 8.4 GB of its 11.8 GB video memory budget in use; another PC without frame
  generation about ten minutes in, whose log names neither the card nor its memory). Not
  reproduced here: two boots with frame generation on a 24 GB card with 8 to 9 GB of it taken
  away ran without one. The next report will carry the driver's fault record (see above).
- With DLSS upscaling on, one of about twenty-five test boots lost the graphics device during
  the load into a save (the record above); four more boots of the same setting did not. Not
  explained, and not known to be DLSS's doing: the next record says which submission held
  DLSS's commands.
- A shader that reads the resident texture table is another compiled variant than before. It is
  built at first use in each part of the game and cached from then on.
- CPU writes into memory that a rendered image also uses still cause whole-image uploads (about
  75 MiB a frame at 100% on Saros).
- Frame generation, the launcher app and the run scripts are Windows only, and so is reading the
  shaders out of a game's files: `ooz_decode`, the program that unpacks Unreal's shader
  libraries for it, is built for Windows only. It is a separate program under GPL-3.0-or-later
  (the ooz decompressor by Powzix), started by the precompile pass and never linked with the
  emulator (`tools/ooz-decode/README.md`).
